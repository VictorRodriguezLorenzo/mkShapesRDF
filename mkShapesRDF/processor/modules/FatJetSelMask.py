import ROOT
from mkShapesRDF.processor.framework.module import Module


class FatJetSelMask(Module):
    def __init__(self, jetId=2, minPt=200., maxEta=2.4, doMask=True, lepton_min_pt=10., DeltaRlep=0.8, DeltaRjet=0.8):
        super().__init__("FatJetSelMask")
        self.jetId = jetId
        self.minPt = minPt
        self.maxEta = maxEta
        self.doMask = doMask
        self.lepton_min_pt = lepton_min_pt
        self.DeltaRlep = DeltaRlep
        self.DeltaRjet = DeltaRjet

    def runModule(self, df, values):
        # This is AK8 selection, NOT the AK4 jet-veto-map event filter.
        ROOT.gInterpreter.Declare(r"""
        #ifndef MKSHAPES_AK8_OVERLAP
        #define MKSHAPES_AK8_OVERLAP
        #include <ROOT/RVec.hxx>
        #include <cmath>
        namespace mkShapesAK8 {
          template<class A, class B, class C, class D>
          ROOT::RVecI separated(const A& eta, const B& phi, const C& otherEta, const D& otherPhi, double radius) {
            ROOT::RVecI result(eta.size(), 1);
            for (size_t i = 0; i < eta.size(); ++i)
              for (size_t j = 0; j < otherEta.size(); ++j) {
                const double dphi = std::remainder(double(phi[i]) - double(otherPhi[j]), 2. * std::acos(-1.));
                const double deta = double(eta[i]) - double(otherEta[j]);
                if (deta*deta + dphi*dphi < radius*radius) { result[i] = 0; break; }
              }
            return result;
          }
        }
        #endif
        """)

        df = df.Define("_fatjet_leptonMask", f"Lepton_pt >= {float(self.lepton_min_pt)}", excludeVariations=["*"])
        if self.doMask:
            selection = (
                f"CorrectedFatJet_pt >= {float(self.minPt)} && abs(CorrectedFatJet_eta) <= {float(self.maxEta)}"
                f" && ((Take(FatJet_jetId, CorrectedFatJet_jetIdx) & {int(self.jetId)}) == {int(self.jetId)})"
                " && mkShapesAK8::separated(CorrectedFatJet_eta, CorrectedFatJet_phi, "
                f"Lepton_eta[_fatjet_leptonMask], Lepton_phi[_fatjet_leptonMask], {float(self.DeltaRlep)})"
            )
        else:
            selection = "ROOT::RVecI(CorrectedFatJet_pt.size(), 1)"
        df = df.Define("_fatjet_mask", selection)

        values.append([df.Define("_fatjet_before", "int(CorrectedFatJet_pt.size())").Sum("_fatjet_before"), "Original size of CorrectedFatJet"])

        df = df.Define("CleanFatJet_jetIdx", "CorrectedFatJet_jetIdx[_fatjet_mask]")
        df = df.Define("CleanFatJet_correctedjetIdx", "ROOT::VecOps::Range(CorrectedFatJet_pt.size())[_fatjet_mask]")
        for prop in ("pt", "eta", "phi", "mass", "msoftdrop"):
            df = df.Define(f"CleanFatJet_{prop}", f"CorrectedFatJet_{prop}[_fatjet_mask]")

        # Keep attributes aligned using the original NanoAOD FatJet indices.
        for col in list(df.GetColumnNames()):
            if col.startswith("FatJet_") and "RVec" in str(df.GetColumnType(col)):
                prop = col[len("FatJet_"):]
                if prop not in ("pt", "eta", "phi", "mass", "msoftdrop"):
                    df = df.Define(f"CleanFatJet_{prop}", f"Take({col}, CleanFatJet_jetIdx)")

        df = df.Define("nCleanFatJet", "int(CleanFatJet_pt.size())")
        values.append([df.Define("_fatjet_after", "int(CleanFatJet_pt.size())").Sum("_fatjet_after"), "Final size of CleanFatJet"])

        cols = set(str(x) for x in df.GetColumnNames())
        if "CleanFatJet_tau1" in cols and "CleanFatJet_tau2" in cols:
            df = df.Define("CleanFatJet_tau21", "ROOT::VecOps::Where(CleanFatJet_tau1 > 0.f, CleanFatJet_tau2 / ROOT::VecOps::Where(CleanFatJet_tau1 > 0.f, CleanFatJet_tau1, 1.f), -1.f)")
        if "CleanJet_eta" in cols and "CleanJet_phi" in cols:
            df = df.Define("CleanJet_notFatJet", f"mkShapesAK8::separated(CleanJet_eta, CleanJet_phi, CleanFatJet_eta, CleanFatJet_phi, {float(self.DeltaRjet)})")
        df = df.Define("Lepton_notFatJet", f"mkShapesAK8::separated(Lepton_eta, Lepton_phi, CleanFatJet_eta, CleanFatJet_phi, {float(self.DeltaRlep)})")
        for kind, branch in (("Muon", "muonIdx"), ("Electron", "electronIdx")):
            if f"Lepton_{branch}" in cols:
                df = df.Define(f"{kind}NotFat_Idx", f"Lepton_{branch}[Lepton_notFatJet && (Lepton_{branch} >= 0)]")
        return df.DropColumns("_fatjet_*")
