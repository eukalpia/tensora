# Fuzzing

This directory owns repository-level fuzz targets, corpus policy, and fuzzing documentation.

The executable native fuzz gate currently exercises the public C ABI under sanitizer instrumentation. Future fuzz targets should be added here or migrated here together with their build wiring so fuzzing remains a first-class release gate rather than an ad-hoc test.
