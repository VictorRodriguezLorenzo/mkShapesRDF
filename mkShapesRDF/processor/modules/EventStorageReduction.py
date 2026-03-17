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
        temporary_columns = []

        def define_if_missing(dataframe, name, expression):
            if name not in column_names:
                dataframe = dataframe.Define(name, expression)
                column_names.append(name)
                temporary_columns.append(name)
            return dataframe

        df = define_if_missing(df, self.bjet_idx_column, 'CleanJet_pt.size() > 0 ? 0 : -1')

        df = define_if_missing(
            df,
            'MTb',
            f'{self.bjet_idx_column} >= 0 ? '
            f'sqrt(2 * CleanJet_pt[{self.bjet_idx_column}] * PuppiMET_pt '
            f'* (1 - cos(DeltaPhi(CleanJet_phi[{self.bjet_idx_column}], PuppiMET_phi)))) : -9999.0',
        )

        df = define_if_missing(
            df,
            'mT2',
            'Lepton_pt.size() > 1 ? '
            'computeMT2(Lepton_pt[0], Lepton_eta[0], Lepton_phi[0], '
            'Lepton_pt[1], Lepton_eta[1], Lepton_phi[1], PuppiMET_pt, PuppiMET_phi) : -9999.0',
        )

        df = define_if_missing(df, 'noJetInHorn', 'Jet_inHorns(CleanJet_pt, CleanJet_eta)')
        df = define_if_missing(df, 'numberCleanJets', 'Sum(CleanJet_pt > 20.)')

        mc_postselection = (
            '((Lepton_pt[0] > 35 && abs(Lepton_pdgId[0]) == 11) || '
            '(Lepton_pt[0] > 30 && abs(Lepton_pdgId[0]) == 13))'
            ' && abs(Lepton_eta[0]) < 2.4'
            ' && mll > 20'
            ' && noJetInHorn'
            ' && numberCleanJets >= 2'
            ' && PuppiMET_pt >= 220'
            ' && MTb > 120'
            ' && ((mtw1 >= 120) || (mT2 <= 80))'
        )

        df = df.Filter(mc_postselection)

        for column in temporary_columns:
            df = df.DropColumns(column)

        return df
