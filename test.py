# -*- coding: utf-8 -*-
# FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
import unittest

def main() -> None:
    suite = unittest.defaultTestLoader.discover("./tests", pattern="*.py")
    if suite.countTestCases() == 0:
        raise RuntimeError("No regression tests were discovered")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)

if __name__ == "__main__":
    main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
