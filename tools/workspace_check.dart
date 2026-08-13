import 'dart:io';

import 'package:yaml/yaml.dart';

const Map<String, String> _workspacePackages = <String, String>{
  'tensora': 'packages/tensora',
  'tensora_nn': 'packages/tensora_nn',
  'tensora_optim': 'packages/tensora_optim',
  'tensora_data': 'packages/tensora_data',
  'tensora_train': 'packages/tensora_train',
  'tensora_edge': 'packages/tensora_edge',
  'tensora_vision': 'packages/tensora_vision',
  'tensora_audio': 'packages/tensora_audio',
  'tensora_text': 'packages/tensora_text',
  'tensora_transformers': 'packages/tensora_transformers',
};

const Map<String, Set<String>> _allowedDependencies = <String, Set<String>>{
  'tensora': <String>{},
  'tensora_nn': <String>{'tensora'},
  'tensora_optim': <String>{'tensora'},
  'tensora_data': <String>{},
  'tensora_train': <String>{'tensora', 'tensora_nn', 'tensora_optim'},
  'tensora_edge': <String>{'tensora'},
  'tensora_vision': <String>{'tensora'},
  'tensora_audio': <String>{'tensora'},
  'tensora_text': <String>{},
  'tensora_transformers': <String>{'tensora_nn', 'tensora_text'},
};

Never _fail(String message) => throw StateError(message);

Map<Object?, Object?> _asMap(Object? value, String context) {
  if (value is! YamlMap) {
    _fail('$context must be a YAML map');
  }
  return value.cast<Object?, Object?>();
}

void main() {
  final root = Directory.current;
  final requiredDirectories = <String>{
    'native',
    'compiler',
    'tools',
    'examples',
    'benchmarks',
    'fuzz',
    'docs',
    'tests',
    ..._workspacePackages.values,
    'packages/tensora_flutter',
  };

  for (final path in requiredDirectories) {
    if (!Directory('${root.path}${Platform.pathSeparator}$path').existsSync()) {
      _fail('Required repository directory is missing: $path');
    }
  }

  final rootPubspec = _asMap(
    loadYaml(File('${root.path}${Platform.pathSeparator}pubspec.yaml').readAsStringSync()),
    'root pubspec',
  );
  final workspace = rootPubspec['workspace'];
  if (workspace is! YamlList) {
    _fail('root workspace must be a YAML list');
  }
  final workspacePaths = workspace.map((value) => value.toString()).toList();
  final actualWorkspace = workspacePaths.toSet();
  final expectedWorkspace = _workspacePackages.values.toSet();
  if (actualWorkspace.length != workspacePaths.length ||
      actualWorkspace.length != expectedWorkspace.length ||
      !actualWorkspace.containsAll(expectedWorkspace)) {
    _fail('root workspace membership does not match the architecture contract');
  }

  final graph = <String, Set<String>>{};
  for (final entry in _workspacePackages.entries) {
    final package = entry.key;
    final directory = Directory('${root.path}${Platform.pathSeparator}${entry.value}');
    final pubspec = _asMap(
      loadYaml(File('${directory.path}${Platform.pathSeparator}pubspec.yaml').readAsStringSync()),
      '$package pubspec',
    );

    if (pubspec['name'] != package) {
      _fail('${entry.value}/pubspec.yaml must declare name: $package');
    }
    if (pubspec['resolution'] != 'workspace') {
      _fail('$package must declare resolution: workspace');
    }

    final internalDependencies = <String>{};
    for (final section in <String>['dependencies', 'dev_dependencies']) {
      final value = pubspec[section];
      if (value == null) continue;
      final dependencies = _asMap(value, '$package $section');
      for (final key in dependencies.keys) {
        final dependency = key.toString();
        if (_workspacePackages.containsKey(dependency)) {
          internalDependencies.add(dependency);
        }
      }
    }

    final forbidden = internalDependencies.difference(
      _allowedDependencies[package]!,
    );
    if (forbidden.isNotEmpty) {
      _fail(
        '$package has forbidden Tensora dependencies: '
        '${forbidden.toList()..sort()}',
      );
    }
    graph[package] = internalDependencies;

    final lib = Directory('${directory.path}${Platform.pathSeparator}lib');
    if (!lib.existsSync()) {
      _fail('$package is missing lib/');
    }
    for (final entity in lib.listSync(recursive: true)) {
      if (entity is! File || !entity.path.endsWith('.dart')) continue;
      final content = entity.readAsStringSync();
      final implementationImport = RegExp(
        r'package:(tensora(?:_[a-z_]+)?)/src/',
      );
      for (final match in implementationImport.allMatches(content)) {
        final importedPackage = match.group(1)!;
        if (importedPackage != package) {
          _fail(
            '${entity.path} imports another package implementation: '
            '$importedPackage/src',
          );
        }
      }
      if (package != 'tensora' &&
          (content.contains("'dart:ffi'") || content.contains('"dart:ffi"'))) {
        _fail('${entity.path} binds dart:ffi outside the tensora foundation');
      }
    }
  }

  final visiting = <String>{};
  final visited = <String>{};

  void visit(String package) {
    if (visited.contains(package)) return;
    if (!visiting.add(package)) {
      _fail('dependency cycle detected at $package');
    }
    for (final dependency in graph[package]!) {
      visit(dependency);
    }
    visiting.remove(package);
    visited.add(package);
  }

  for (final package in graph.keys) {
    visit(package);
  }

  final flutterPubspec = File(
    '${root.path}${Platform.pathSeparator}packages'
    '${Platform.pathSeparator}tensora_flutter'
    '${Platform.pathSeparator}pubspec.yaml',
  );
  if (!flutterPubspec.existsSync()) {
    _fail('tensora_flutter is missing pubspec.yaml');
  }

  stdout.writeln(
    'Workspace contract OK: ${_workspacePackages.length} Dart packages, '
    '1 Flutter package, acyclic dependencies.',
  );
}
