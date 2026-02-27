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
            "addGeomInfo",
            "addVarInfo",
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
            "Decode",
            "DecodeGeom",
            "DecodeVar",
            "DecodeAtGrid",
            "AbsPointSetError",
            "FixedEncode",
            "FixedEncodeGeom",
            "FixedEncodeVar",
            "AdaptiveEncode",
            "AdaptiveEncodeGeom",
            "AdaptiveEncodeVar",
            "RayEncode",
            "IntegratePointSet",
            "AddGeometry",
            "AddVariable",
            "geom",
            "var",
            "shiftGeom",
            "shiftVar",
            "printDetails",
        ]
        required_pointset = [
            "validate",
            "is_same_layout",
            "is_structured",
            "nvars",
            "geom_dim",
            "var_dim",
            "var_min",
            "var_max",
            "ndom_pts",
            "model_dims",
            "set_bounds",
            "set_domain",
            "set_domain_params",
            "set_grid_params",
            "mins",
            "maxs",
            "abs_diff",
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

    def test_mfainfo_add_var_vector_overload(self):
        dom_dim = 2
        geom = self.mfa.ModelInfo(dom_dim, dom_dim, 1, 2)
        var0 = self.mfa.ModelInfo(dom_dim, 1, 1, 2)
        var1 = self.mfa.ModelInfo(dom_dim, 2, 1, 2)

        info = self.mfa.MFAInfo(dom_dim, 0)
        info.addGeomInfo(geom)
        info.addVarInfo([var0, var1])

        self.assertEqual(info.nvars(), 2)
        self.assertEqual(info.var_dim(0), 1)
        self.assertEqual(info.var_dim(1), 2)
        self.assertEqual(info.pt_dim(), 5)

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

    def test_mfa_modelinfo_overload_construction(self):
        dom_dim = 2
        model = self.mfa.MFA(dom_dim, 0)

        geom = self.mfa.ModelInfo(dom_dim, dom_dim, 1, 2)
        var = self.mfa.ModelInfo(dom_dim, 1, 1, 2)

        model.AddGeometry(geom)
        model.AddVariable(var)

        self.assertEqual(model.nvars(), 1)
        self.assertEqual(model.geom_dim(), 2)
        self.assertEqual(model.var_dim(0), 1)

    def test_pointset_constructor_accepts_lists_and_numpy(self):
        pts_list = self.mfa.PointSet(2, [2, 1], 4, [2, 2])
        self.assertEqual(pts_list.dom_dim, 2)
        self.assertEqual(pts_list.pt_dim, 3)
        self.assertEqual(pts_list.npts, 4)

        pts_numpy = self.mfa.PointSet(
            2,
            np.array([2, 1], dtype=np.int32),
            4,
            np.array([2, 2], dtype=np.int32),
        )
        self.assertEqual(pts_numpy.dom_dim, 2)
        self.assertEqual(pts_numpy.pt_dim, 3)
        self.assertEqual(pts_numpy.npts, 4)

    def test_pointset_layout_helpers_without_domain_assignment(self):
        pts = self.mfa.PointSet(2, [2, 1], 4, [2, 2])
        same_layout = self.mfa.PointSet(2, [2, 1], 4, [2, 2])
        different_layout = self.mfa.PointSet(2, [2, 2], 4, [2, 2])

        self.assertTrue(pts.validate())
        self.assertTrue(pts.is_structured())
        self.assertEqual(pts.nvars(), 1)
        self.assertEqual(pts.geom_dim(), 2)
        self.assertEqual(pts.var_dim(0), 1)
        self.assertEqual(pts.var_min(0), 2)
        self.assertEqual(pts.var_max(0), 2)
        self.assertEqual(pts.ndom_pts(0), 2)
        self.assertEqual(pts.ndom_pts(1), 2)
        np.testing.assert_array_equal(np.asarray(pts.ndom_pts()), np.array([2, 2]))
        np.testing.assert_array_equal(np.asarray(pts.model_dims()), np.array([2, 1]))
        self.assertTrue(pts.is_same_layout(same_layout, 0))
        self.assertFalse(pts.is_same_layout(different_layout, 0))

    def test_domain_args_helpers(self):
        d_args = self.mfa.DomainArgs(2, [2, 1])
        d_args.updateModelDims([2, 2, 1])
        self.assertEqual(len(d_args.s), 2)
        self.assertEqual(len(d_args.f), 2)


if __name__ == "__main__":
    unittest.main()
