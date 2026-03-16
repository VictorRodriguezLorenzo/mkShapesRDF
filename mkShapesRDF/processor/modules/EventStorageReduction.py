from mkShapesRDF.processor.framework.module import Module
import ROOT


class EventStorageReduction(Module):
    def __init__(
        self,
        mt2_include_path='/afs/cern.ch/user/v/victorr/private/PlotsConfigurationsRun3/topDM/TTDMsimp_semileptonic/macros/computeMT2.cc',
        jet_horns_include_path='/afs/cern.ch/user/v/victorr/private/PlotsConfigurationsRun3/topDM/TTDMsimp_semileptonic/extended/jet_horns.cc',
        bjet_idx_column='bjet_idx',
    ):
        super().__init__('EventStorageReduction')
        self.mt2_include_path = mt2_include_path
        self.jet_horns_include_path = jet_horns_include_path
        self.bjet_idx_column = bjet_idx_column

    def runModule(self, df, values):
        ROOT.gInterpreter.Declare(f'#include "{self.mt2_include_path}"')
        ROOT.gInterpreter.Declare(f'#include "{self.jet_horns_include_path}"')

        column_names = [str(c) for c in df.GetColumnNames()]

        if self.bjet_idx_column not in column_names:
            df = df.Define(self.bjet_idx_column, 'CleanJet_pt.size() > 0 ? 0 : -1')

        if 'mll' not in column_names:
            df = df.Define(
                'mll',
                'Lepton_pt.size() > 1 ? '
                '(ROOT::Math::PtEtaPhiMVector(Lepton_pt[0], Lepton_eta[0], Lepton_phi[0], 0) '
                '+ ROOT::Math::PtEtaPhiMVector(Lepton_pt[1], Lepton_eta[1], Lepton_phi[1], 0)).M() : -9999.0',
            )

        if 'mtw1' not in column_names:
            df = df.Define(
                'mtw1',
                'Lepton_pt.size() > 0 ? sqrt(2. * Lepton_pt[0] * PuppiMET_pt '
                '* (1. - cos(DeltaPhi(Lepton_phi[0], PuppiMET_phi)))) : -9999.0',
            )

        df = df.Define(
            'MTb',
            f'{self.bjet_idx_column} >= 0 ? '
            f'sqrt(2 * CleanJet_pt[{self.bjet_idx_column}] * PuppiMET_pt '
            f'* (1 - cos(DeltaPhi(CleanJet_phi[{self.bjet_idx_column}], PuppiMET_phi)))) : -9999.0',
        )

        df = df.Define(
            'mT2',
            'Lepton_pt.size() > 1 ? '
            'computeMT2(Lepton_pt[0], Lepton_eta[0], Lepton_phi[0], '
            'Lepton_pt[1], Lepton_eta[1], Lepton_phi[1], PuppiMET_pt, PuppiMET_phi) : -9999.0',
        )

        df = df.Define('noJetInHorn', 'Jet_inHorns(CleanJet_pt, CleanJet_eta)')
        df = df.Define('njets', 'Sum(CleanJet_pt > 20.)')

        mc_postselection = (
            '((Lepton_pt[0] > 35 && abs(Lepton_pdgId[0]) == 11) || '
            '(Lepton_pt[0] > 30 && abs(Lepton_pdgId[0]) == 13))'
            ' && abs(Lepton_eta[0]) < 2.4'
            ' && mll > 20'
            ' && noJetInHorn'
            ' && njets >= 2'
            ' && PuppiMET_pt >= 220'
            ' && MTb > 120'
            ' && ((mtw1 >= 120) || (mT2 <= 80))'
        )

        df = df.Filter(mc_postselection)
        return df

