import os

Sites = {

    'cern' : {
        "eosTmpWorkDir": "'USEDAS'", ### Optional to redirect the input samples
        #"eosDir"       : "/eos/cms/store/group/phys_higgs/cmshww/calderon/HWWNano/",
        "eosDir"       : "/eos/user/v/victorr/HWWNano/",
        "redirector"    : "root://cmsxrootd.fnal.gov/", # xrootd-cms.infn.it. # cmsxrootd.fnal.gov
    },

    'kit' : {
        "eosTmpWorkDir": "'/tmp/'", # + os.getlogin() + "/'", ### Optional to redirect the input samples
        "eosDir"       : "/ceph/ntrevisa/HWWNano/",
        "redirector"    : "root://cmsxrootd.fnal.gov/", # xrootd-cms.infn.it. # cmsxrootd.fnal.gov
    }
}
