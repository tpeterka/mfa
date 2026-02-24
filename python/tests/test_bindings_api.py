import importlib
import unittest

import numpy as np


class TestBindingsAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.mfa = importlib.import_module("mfa")
        except ModuleNotFoundError as exc:
            raise unittest.SkipTest(f"mfa module unavailable: {exc}") from exc

    def _build_info(self):
        dom_dim = 2
        geom = self.mfa.ModelInfo(dom_dim, dom_dim, 1, 2)
        var = self.mfa.ModelInfo(dom_dim, 1, 1, 2)
        info = self.mfa.MFAInfo(dom_dim, 0)
        info.addGeomInfo(geom)
        info.addVarInfo(var)
        return info

    def test_new_api_surface_is_exported(self):
        required_mfainfo = [
            "nvars",
            "geom_dim",
            "pt_dim",
            "var_dim",
            "model_dims",
            "splitStrongScaling",
            "reset",
        ]
        required_mfa = [
            "nvars",
            "geom_dim",
            "var_dim",
            "model_dims",
            "setGeomKnots",
            "setKnots",
            "DefiniteIntegral",
            "Integrate1D",
            "DecodeGeom",
            "DecodeVar",
            "AbsPointSetError",
        ]
        required_pointset = [
            "pt_coords",
            "geom_coords",
            "var_coords",
            "pt_params",
            "validate",
            "is_same_layout",
        ]

        for name in required_mfainfo:
            self.assertTrue(hasattr(self.mfa.MFAInfo, name), name)

        for name in required_mfa:
            self.assertTrue(hasattr(self.mfa.MFA, name), name)

        for name in required_pointset:
            self.assertTrue(hasattr(self.mfa.PointSet, name), name)

    def test_mfainfo_helper_methods(self):
        info = self._build_info()

        self.assertEqual(info.nvars(), 1)
        self.assertEqual(info.geom_dim(), 2)
        self.assertEqual(info.var_dim(0), 1)
        self.assertEqual(info.pt_dim(), 3)
        np.testing.assert_array_equal(np.asarray(info.model_dims()), np.array([2, 1]))

    def test_mfa_helpers_work_for_basic_cases(self):
        info = self._build_info()
        model = self.mfa.MFA(info)

        self.assertEqual(model.nvars(), 1)
        self.assertEqual(model.geom_dim(), 2)
        self.assertEqual(model.var_dim(0), 1)
        np.testing.assert_array_equal(np.asarray(model.model_dims()), np.array([2, 1]))

        model.setGeomKnots([])
        model.setKnots([])
        model.setKnots(0, [])


if __name__ == "__main__":
    unittest.main()
