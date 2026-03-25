from mkShapesRDF.processor.framework.module import Module
import ROOT


class EventStorageReduction(Module):
    def __init__(
        self,
        jet_horns_include_path='/afs/cern.ch/user/v/victorr/private/PlotsConfigurationsRun3/topDM/TTDMsimp_semileptonic/extended/jet_horns.cc',
    ):
        super().__init__('EventStorageReduction')
        self.jet_horns_include_path = jet_horns_include_path

    def runModule(self, df, values):
        ROOT.gInterpreter.Declare(f'#include "{self.jet_horns_include_path}"')

        column_names = [str(c) for c in df.GetColumnNames()]
        temporary_columns = []

        def define_if_missing(dataframe, name, expression):
            if name not in column_names:
                dataframe = dataframe.Define(name, expression)
                column_names.append(name)
                temporary_columns.append(name)
            return dataframe

        df = define_if_missing(df, 'noJetInHorn', 'Jet_inHorns(CleanJet_pt, CleanJet_eta)')
        df = define_if_missing(df, 'numberCleanJets', 'Sum(CleanJet_pt > 30.)')

        mc_postselection = (
            '((Lepton_pt[0] > 35 && abs(Lepton_pdgId[0]) == 11) || '
            '(Lepton_pt[0] > 30 && abs(Lepton_pdgId[0]) == 13))'
            ' && abs(Lepton_eta[0]) < 2.4'
            ' && numberCleanJets >= 2'
            ' && PuppiMET_pt >= 150'
        )

        df = df.Filter(mc_postselection)

        for column in temporary_columns:
            df = df.DropColumns(column)

        return df
