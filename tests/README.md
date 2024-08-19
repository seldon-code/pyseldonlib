# Tests

This is the main directory for the pyseldon bindings and package tests.

These are just the same as the tests in the main seldon repository, but are run in the context of the package in Python.

the seldoncore directory tests the bindings to the seldon code repository, and allother tests the package itself.

Currently only the activity_driven test for 10 agent meanfield is failing for the time being I have disabled it in the tests. Because it seems to be a upstream issue with the seldon codebase to me