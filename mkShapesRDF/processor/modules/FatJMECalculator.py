import re
import json
import uuid
from pathlib import Path

import ROOT

from mkShapesRDF.processor.framework.module import Module
from mkShapesRDF.processor.data.JetMaker_cfg import JetMakerCfg

FATJET_SUBJET_JES_CPP = r'''
#ifndef MKSHAPES_FATJET_SUBJET_JES
#define MKSHAPES_FATJET_SUBJET_JES
#include "FatJetVariationsCalculator.h"
#include <Math/GenVector/LorentzVector.h>
#include <Math/GenVector/PtEtaPhiM4D.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <cstdint>

class MkShapesFatJetVariationsCalculator : public FatJetVariationsCalculator {
  class SubJetSmearer : public JetMETVariationsCalculatorBase {
  public:
    void configure(const std::string& json, const std::string& tag,
                   const std::string& algorithm, const std::string& smearJson,
                   bool match) {
      auto corrections = correction::CorrectionSet::from_file(json);
      auto smearing = correction::CorrectionSet::from_file(smearJson);
      const auto suffix = "_" + algorithm;
      const auto uncertainty = tag + "_SFUncertainty" + suffix;
      const bool newFormat = hasCorrection(corrections, uncertainty);
      setSmearing(corrections->at(tag + "_PtResolution" + suffix),
                  corrections->at(tag + "_ScaleFactor" + suffix),
                  newFormat ? corrections->at(uncertainty) : nullptr,
                  smearing->at("JERSmear"), false, newFormat, match, 0.2, 3.0);
      checkSmearingArguments();
    }
    std::array<double, 3> factors(double pt, float eta, float phi,
        const p4compv_t& genpt, const p4compv_t& geneta, const p4compv_t& genphi,
        int seed, float rho) const {
      if (pt <= 0.) return {1., 1., 1.};
      // The pinned backend's unsigned findGenMatch index cannot safely accept
      // -1. Associate explicitly and bypass its indexed lookup when unmatched.
      int match = -1;
      double bestDR2 = 0.04;
      if (m_smearDoGenMatch) {
        const auto resolution = m_jetPtRes->evaluate({double(eta), pt, double(rho)});
        for (std::size_t i = 0; i < genpt.size(); ++i) {
          const double dphi = std::remainder(double(phi) - genphi[i], 2. * M_PI);
          const double deta = double(eta) - geneta[i];
          const double dr2 = dphi * dphi + deta * deta;
          if (dr2 < bestDR2 && std::abs(pt - genpt[i]) < 3. * resolution * pt) {
            bestDR2 = dr2;
            match = int(i);
          }
        }
      }
      return applyJERSmearing(m_jetPtRes, m_jerSmear, m_jetEResSF,
          m_jetEResSFUnc, match >= 0, pt, eta, phi, match >= 0 ? match : 0,
          genpt, geneta, genphi, seed, rho);
    }
  };
  class AK8OnlyCalculator : public FatJetVariationsCalculator {
  public:
    explicit AK8OnlyCalculator(FatJetVariationsCalculator base)
        : FatJetVariationsCalculator(std::move(base)) { m_doJECSubjet = false; }
  };
  enum class Input {Area, Eta, Pt, Rho, Phi, Run};
  struct InputSpec {
    Input input;
    correction::Variable::VarType type;
  };
  using Schema = std::vector<InputSpec>;
  using LorentzVector = ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>;

  Schema m_subjetJECInputs;
  std::map<std::string, Schema> m_subjetJESInputs;
  AK8OnlyCalculator m_ak8Only;
  SubJetSmearer m_subjetSmearer;
  bool m_smearSubjets = false;

  static Schema inputSchema(const std::vector<correction::Variable>& inputs) {
    const std::map<std::string, Input> known = {
      {"JetA", Input::Area}, {"JetEta", Input::Eta}, {"JetPt", Input::Pt},
      {"Rho", Input::Rho}, {"JetPhi", Input::Phi}, {"run", Input::Run}
    };
    Schema result;
    for (const auto& variable : inputs) {
      const auto found = known.find(variable.name());
      if (found == known.end())
        throw std::invalid_argument("Unsupported SubJet correction input: " + variable.name());
      const auto type = variable.type();
      if (type != correction::Variable::VarType::real &&
          !(found->second == Input::Run && type == correction::Variable::VarType::integer))
        throw std::invalid_argument("Unsupported SubJet correction input type: " + variable.name());
      result.push_back({found->second, type});
    }
    return result;
  }

  static std::vector<correction::Variable::Type> inputValues(
      const Schema& schema, double eta, double pt, double phi, double rho, int run) {
    std::vector<correction::Variable::Type> values;
    for (const auto& variable : schema) {
      double value = 0.;
      switch (variable.input) {
        // The PUPPI SubJet recipe has no area branch and uses area zero.
        case Input::Area: value = 0.; break;
        case Input::Eta: value = eta; break;
        case Input::Pt: value = pt; break;
        case Input::Rho: value = rho; break;
        case Input::Phi: value = phi; break;
        case Input::Run: value = run; break;
      }
      if (variable.type == correction::Variable::VarType::integer)
        values.emplace_back(run);
      else
        values.emplace_back(value);
    }
    return values;
  }

  static ROOT::RVecF groomedMass(
      const p4compv_int& first, const p4compv_int& second,
      const ROOT::RVec<double>& pt, const p4compv_t& eta, const p4compv_t& phi,
      const ROOT::RVec<double>& mass, const ROOT::RVec<double>& factors) {
    ROOT::RVecF result(first.size(), 0.);
    for (std::size_t jet = 0; jet < first.size(); ++jet) {
      if (first[jet] < 0 || second[jet] < 0 ||
          first[jet] >= int(pt.size()) || second[jet] >= int(pt.size())) continue;
      const auto i = first[jet], j = second[jet];
      const auto pair =
        LorentzVector(pt[i] * factors[i], eta[i], phi[i], mass[i] * factors[i]) +
        LorentzVector(pt[j] * factors[j], eta[j], phi[j], mass[j] * factors[j]);
      result[jet] = std::abs(pair.M());
    }
    return result;
  }

  static void setSoftDrop(result_t& result, std::size_t index, const ROOT::RVecF& mass) {
    // Copy the other components before replacing this entry in the result.
    auto pt = result.pt(index);
    auto ungroomed = result.mass(index);
    result.set(index, pt, ungroomed, mass);
  }

public:
  explicit MkShapesFatJetVariationsCalculator(FatJetVariationsCalculator base,
      const std::string& subjetJson = "", const std::string& subjetJER = "",
      const std::string& algorithm = "AK4PFPuppi",
      const std::string& smearJson = "", bool genMatch = true)
      : FatJetVariationsCalculator(base), m_ak8Only(std::move(base)) {
    if (m_addHEM2018Issue)
      throw std::invalid_argument("The Run 3 SubJet adapter does not support HEM variations");
    if (!m_doJECSubjet) return;
    if (!subjetJER.empty()) {
      m_subjetSmearer.configure(subjetJson, subjetJER, algorithm, smearJson, genMatch);
      m_smearSubjets = true;
    }
    m_subjetJECInputs = std::visit([](const auto& corrector) {
      if (!corrector) throw std::invalid_argument("Missing SubJet JEC corrector");
      return inputSchema(corrector->inputs());
    }, m_jesSFSubjet);
    for (const auto& source : m_jesUncSources) {
      const auto corrector = m_jesUncSourcesSubjet.at(source.first);
      m_subjetJESInputs.emplace(source.first, inputSchema(corrector->inputs()));
    }
  }

  result_t produce(
      const p4compv_t& jet_pt, const p4compv_t& jet_eta, const p4compv_t& jet_phi,
      const p4compv_t& jet_mass, const p4compv_t& jet_rawcorr, const p4compv_t& jet_area,
      const p4compv_t& jet_msoftdrop, const p4compv_int& first, const p4compv_int& second,
      const p4compv_t& subpt, const p4compv_t& subeta, const p4compv_t& subphi,
      const p4compv_t& submass, const p4compv_t& subraw,
      const p4compv_int& jet_id, float rho, const p4compv_int& genidx,
      int seed, int run, const p4compv_t& genpt, const p4compv_t& geneta,
      const p4compv_t& genphi, const p4compv_t& genmass,
      const p4compv_t& subgenpt = {}, const p4compv_t& subgeneta = {},
      const p4compv_t& subgenphi = {}) const {
    const auto n = subpt.size();
    if (subeta.size() != n || subphi.size() != n || submass.size() != n || subraw.size() != n)
      throw std::invalid_argument("Unaligned SubJet inputs");
    if (first.size() != jet_pt.size() || second.size() != jet_pt.size())
      throw std::invalid_argument("Unaligned FatJet SubJet indices");
    if (subgeneta.size() != subgenpt.size() || subgenphi.size() != subgenpt.size())
      throw std::invalid_argument("Unaligned SubGenJetAK8 inputs");
    // Keep the native AK8 kinematics, while bypassing its SubJet evaluation.
    // The copy is configured once at construction, rather than for each event.
    // Guard the native unsigned indexed lookup for unmatched AK8 jets. A
    // distant sentinel lets it search and smear without dereferencing -1.
    auto safeGenIdx = genidx;
    auto safeGenPt = genpt, safeGenEta = geneta, safeGenPhi = genphi, safeGenMass = genmass;
    if (m_doSmearing) {
      if (geneta.size() != genpt.size() || genphi.size() != genpt.size() ||
          genmass.size() != genpt.size() || genidx.size() != jet_pt.size())
        throw std::invalid_argument("Unaligned GenJetAK8 inputs");
      bool needsSentinel = false;
      for (auto& index : safeGenIdx)
        if (index < 0 || index >= int(genpt.size())) {
          index = int(genpt.size());
          needsSentinel = true;
        }
      if (needsSentinel) {
        safeGenPt.push_back(0.f); safeGenEta.push_back(1.e6f);
        safeGenPhi.push_back(0.f); safeGenMass.push_back(0.f);
      }
    }
    auto result = m_ak8Only.produce(
      jet_pt, jet_eta, jet_phi, jet_mass, jet_rawcorr, jet_area, jet_msoftdrop,
      first, second, subpt, subeta, subphi, submass, subraw, jet_id, rho,
      safeGenIdx, seed, run, safeGenPt, safeGenEta, safeGenPhi, safeGenMass);
    if (!m_doJECSubjet) return result;

    ROOT::RVec<double> pt(subpt), mass(submass);
    for (std::size_t i = 0; i < n; ++i) {
      const auto values = inputValues(
        m_subjetJECInputs, subeta[i], subpt[i] * (1. - subraw[i]), subphi[i], rho, run);
      const double correction = std::visit([&](const auto& corrector) {
        return corrector->evaluate(values);
      }, m_jesSFSubjet);
      if (!std::isfinite(correction))
        throw std::runtime_error("Nonfinite SubJet JEC");
      if (correction > 0.) {
        const double factor = (1. - subraw[i]) * correction;
        pt[i] *= factor;
        mass[i] *= factor;
      }
    }
    // Scale each corrected SubJet four-vector with its own AK4 JER. Evaluate
    // once per original index, including SubJets shared by multiple FatJets.
    ROOT::RVec<double> jerNom(n, 1.), jerUp(n, 1.), jerDown(n, 1.);
    if (m_smearSubjets) {
      for (std::size_t i = 0; i < n; ++i) {
        const int subjetSeed = int((uint64_t(uint32_t(seed)) +
            0x9e3779b9ULL * (i + 1)) & 0x7fffffffULL);
        const auto factors = m_subjetSmearer.factors(
            pt[i], subeta[i], subphi[i], subgenpt, subgeneta, subgenphi, subjetSeed, rho);
        for (auto factor : factors)
          if (!std::isfinite(factor) || factor < 0.)
            throw std::runtime_error("Invalid SubJet JER factor");
        jerNom[i] = factors[0]; jerUp[i] = factors[1]; jerDown[i] = factors[2];
      }
    }
    const auto nominal = groomedMass(first, second, pt, subeta, subphi, mass, jerNom);
    const auto labels = available("msoftdrop");
    if (labels.size() != result.size() || labels.size() != result.sizeM())
      throw std::runtime_error("Unsupported FatJet backend result layout");
    std::vector<bool> replaced(labels.size(), false);
    for (std::size_t i = 0; i < labels.size(); ++i)
      if (labels[i] == "nominal") {
        setSoftDrop(result, i, nominal);
        replaced[i] = true;
      } else if (labels[i] == "jerup" || labels[i] == "jerdown") {
        setSoftDrop(result, i, groomedMass(first, second, pt, subeta, subphi,
            mass, labels[i] == "jerup" ? jerUp : jerDown));
        replaced[i] = true;
      }

    for (const auto& source : m_jesUncSourcesSubjet) {
      ROOT::RVec<double> up(jerNom), down(jerNom);
      for (std::size_t i = 0; i < n; ++i) {
        const auto delta = source.second->evaluate(inputValues(
          m_subjetJESInputs.at(source.first), subeta[i], pt[i] * jerNom[i], subphi[i], rho, run));
        if (!std::isfinite(delta)) throw std::runtime_error("Nonfinite SubJet JES");
        up[i] *= 1. + delta;
        down[i] *= 1. - delta;
      }
      for (const auto& direction : {std::string("up"), std::string("down")}) {
        const auto label = "jes" + source.first + direction;
        const auto index = std::find(labels.begin(), labels.end(), label);
        if (index == labels.end())
          throw std::runtime_error("Missing FatJet soft-drop variation: " + label);
        const auto position = std::distance(labels.begin(), index);
        setSoftDrop(result, position, groomedMass(
          first, second, pt, subeta, subphi, mass, direction == "up" ? up : down));
        replaced[position] = true;
      }
    }
    if (std::find(replaced.begin(), replaced.end(), false) != replaced.end())
      throw std::runtime_error("Unsupported FatJet soft-drop variation label");
    return result;
  }
};
#endif
'''

def unique_name(prefix):
    return prefix + uuid.uuid4().hex


def cpp_string(value):
    return json.dumps(str(value))


def require_columns(df, columns):
    missing = set(columns) - set(df.GetColumnNames())
    if missing:
        raise ValueError("Missing NanoAOD columns: " + ", ".join(sorted(missing)))


def period_switch(configs, expressions):
    if None in configs:
        return expressions[None]
    # The comma expression keeps the correct C++ return type for arbitrary objects.
    first = expressions[next(iter(configs))]
    result = f'(throw std::runtime_error("Unknown JME campaign period"), ({first}))'
    for period in reversed(list(configs)):
        result = f'(int(run_period) == {period} ? ({expressions[period]}) : {result})'
    return result



def variation_maps(calculator):
    """Index each observable independently: mass-only variations need not vary pt."""
    maps = {}
    for attr in ("pt", "mass", "msoftdrop"):
        labels = [str(x) for x in calculator.available(attr)]
        if not labels or labels[0] != "nominal" or len(set(labels)) != len(labels):
            raise ValueError(f"Invalid CMSJMECalculators variation labels for {attr}")
        maps[attr] = {label: i for i, label in enumerate(labels)}
    sources = {}
    for labels in maps.values():
        for label in labels:
            if label == "nominal":
                continue
            match = re.fullmatch(r"(.+)(up|down)", label)
            if not match:
                raise ValueError(f"Unsupported JME variation label {label!r}")
            sources.setdefault(match[1], set()).add(match[2])
    if any(tags != {"up", "down"} for tags in sources.values()):
        raise ValueError("JME variations must have both up and down labels")
    for labels in maps.values():
        for source in sources:
            if (source + "up" in labels) != (source + "down" in labels):
                raise ValueError(f"Incomplete JME variation {source}")
    return maps, sorted(sources)


class FatJMECalculator(Module):
    def __init__(
        self,
        jet_object="AK8PFPuppi",
        jes_unc=("Total",),
        year="",
        do_JER=True,
        store_nominal=True,
        store_variations=True,
        isMC=True,
        sampleName="",
        config=None,
        apply_mass_calibration=False,
        jer_variation_name="jer",
    ):
        super().__init__("FatJMECalculator")
        if jet_object != "AK8PFPuppi":
            raise ValueError("Run 3 FatJets require the AK8PFPuppi algorithm")
        self.jet_object = jet_object
        self.jes_unc = tuple(jes_unc)
        self.year = year
        self.isMC = isMC
        self.do_JER = do_JER
        self.store_nominal = store_nominal
        self.store_variations = store_variations
        if not self.isMC:
            self.do_JER = False
            self.store_variations = False
        self.sampleName = sampleName
        self.apply_mass_calibration = bool(apply_mass_calibration and isMC)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", jer_variation_name):
            raise ValueError("jer_variation_name must be a valid variation identifier")
        self.jer_variation_name = jer_variation_name
        campaign = JetMakerCfg[year] if config is None else config
        self.runPeriods = [k for k in campaign if str(k).isdigit()]
        self.configs = ({int(k): campaign[k] for k in self.runPeriods}
                        if self.runPeriods else {None: campaign})

    def _make_calculator(self, cfg):
        if self.apply_mass_calibration:
            raise ValueError(
                "The Run 3 soft-drop recipe applies no additional central JMS/JMR correction; use apply_mass_calibration=False"
            )
        settings = cfg["fat_jet"]
        jet_settings = cfg["jet"]
        required = ("json", "subjet_algorithm", "jec_level", "subjet_jec_level")
        for key in required:
            if key not in settings:
                raise ValueError(f"Missing fat_jet configuration: {key}")
        for key in ("JEC" if self.isMC else "JEC_data", "JER"):
            if key not in cfg:
                raise ValueError(f"Missing shared campaign correction: {key}")
        if "jet_jerc" not in jet_settings:
            raise ValueError("Missing AK4 jet_jerc payload for subjets")
        if self.do_JER and "jer_smear" not in cfg:
            raise ValueError("Missing shared campaign jer_smear payload")
        subjet_json = jet_settings["jet_jerc"]
        for path in (settings["json"], subjet_json) + ((cfg["jer_smear"],) if self.do_JER else ()):
            if not Path(path).is_file():
                raise FileNotFoundError(f"Missing FatJet payload: {path}")
        tag = cfg["JEC" if self.isMC else "JEC_data"]
        if isinstance(tag, list):
            if not self.isMC and len(tag) == 3:
                match = re.search(r"Run2022([EFG])", self.sampleName)
                if match:
                    tag = tag["EFG".index(match[1])]
            elif not self.isMC and len(tag) == 2:
                match = re.search(r"(?:^|[-_])v([1-4])(?:$|[-_])", self.sampleName)
                if match:
                    tag = tag[int(match[1]) == 4]
            if not isinstance(tag, str):
                raise ValueError(f"Cannot resolve data JEC tag for {self.sampleName!r}")
        subjet_tag = tag
        match = re.search(r"20\d{2}", self.year)
        uncertainty_year = str(cfg.get("year", match.group(0) if match else ""))
        uncertainties = ROOT.std.vector("string")()
        for source in self.jes_unc if self.store_variations else ():
            if "YEAR" in source and not uncertainty_year:
                raise ValueError("Cannot resolve the year for JES uncertainties")
            uncertainties.push_back(source.replace("YEAR", uncertainty_year))
        name = unique_name("fatjet_calculator_")
        jer_tag = cfg["JER"] if self.do_JER else ""
        subjet_jer = jer_tag
        args = [
            cpp_string(settings["json"]),
            cpp_string(self.jet_object),
            cpp_string(tag),
            cpp_string(settings["jec_level"]),
            "std::vector<std::string>{"
            + ",".join(cpp_string(x) for x in uncertainties)
            + "}",
            "false",
            cpp_string(jer_tag),
            cpp_string(cfg["jer_smear"] if self.do_JER else ""),
            cpp_string("JERSmear" if self.do_JER else ""),
            # No HEM-2018 recipe, no split JER; hybrid match DeltaR < R/2 and 3 sigma.
            "false",
            "true",
            "0.4",
            "3.0",
            cpp_string(subjet_json),
            cpp_string(settings["subjet_algorithm"]),
            cpp_string(subjet_tag),
            cpp_string(settings["subjet_jec_level"]),
        ]
        if not ROOT.gInterpreter.Declare(FATJET_SUBJET_JES_CPP):
            raise RuntimeError("Could not compile FatJet SubJet adapter")
        if not ROOT.gInterpreter.Declare(
            f"auto {name} = MkShapesFatJetVariationsCalculator("
            f"FatJetVariationsCalculator::create({', '.join(args)}), "
            f"{cpp_string(subjet_json)}, {cpp_string(subjet_jer)}, "
            f"{cpp_string(settings['subjet_algorithm'])}, "
            f"{cpp_string(cfg['jer_smear'] if self.do_JER else '')}, "
            f"{'true' if settings.get('subjet_gen_match', True) else 'false'});"
        ):
            raise RuntimeError("Could not construct FatJet calculator")
        calculator = getattr(ROOT, name)
        return name, calculator

    def runModule(self, df, values):
        if any("fat_jet" not in cfg for cfg in self.configs.values()):
            raise ValueError(
                "Configure campaign-specific fatjet payloads for every run period"
            )
        columns = [
            "FatJet_" + x
            for x in (
                "pt",
                "eta",
                "phi",
                "mass",
                "rawFactor",
                "area",
                "msoftdrop",
                "subJetIdx1",
                "subJetIdx2",
                "jetId",
            )
        ]
        columns += ["SubJet_" + x for x in ("pt", "eta", "phi", "mass", "rawFactor")]
        columns += ["Rho_fixedGridRhoFastjetAll"]
        columns += ["run"]
        if self.do_JER:
            columns += ["FatJet_genJetAK8Idx", "run", "luminosityBlock", "event"]
            columns += ["GenJetAK8_" + x for x in ("pt", "eta", "phi", "mass")]
            if any(
                cfg["fat_jet"].get("subjet_gen_match", True)
                for cfg in self.configs.values()
            ):
                columns += ["SubGenJetAK8_" + x for x in ("pt", "eta", "phi")]
        if None not in self.configs:
            columns += ["run_period"]
        require_columns(df, columns)
        from CMSJMECalculators import loadJMESystematicsCalculators

        loadJMESystematicsCalculators()
        if not hasattr(getattr(ROOT, "FatJetVariationsCalculator", None), "create"):
            raise RuntimeError(
                "The correctionlib Run 3 CMSJMECalculators backend is required"
            )
        args = columns[:9] + [
            "SubJet_" + x for x in ("pt", "eta", "phi", "mass", "rawFactor")
        ]
        args += ["FatJet_jetId", "Rho_fixedGridRhoFastjetAll"]
        # NanoAOD stores some index/ID branches as short or unsigned char.
        for index in (7, 8, 14):
            branch = args[index]
            args[index] = f"ROOT::RVecI({branch}.begin(), {branch}.end())"
        if self.do_JER:
            # unsigned arithmetic avoids undefined shifts on signed NanoAOD run values.
            args += [
                "ROOT::RVecI(FatJet_genJetAK8Idx.begin(), FatJet_genJetAK8Idx.end())",
                "int(((ULong64_t(run)<<20) + (ULong64_t(luminosityBlock)<<10) + ULong64_t(event)) & 0x7fffffffULL)",
                "int(run)",
            ]
            args += ["GenJetAK8_" + x for x in ("pt", "eta", "phi", "mass")]
        else:
            args += ["ROOT::RVecI{}", "0", "int(run)"] + ["ROOT::RVecF{}"] * 4
        if self.do_JER:
            if all("SubGenJetAK8_" + x in columns for x in ("pt", "eta", "phi")):
                args += ["SubGenJetAK8_" + x for x in ("pt", "eta", "phi")]
            else:
                args += ["ROOT::RVecF{}"] * 3
        # subJetIdx1/2 are original SubJet indices; never sort or preselect that collection.
        # The adapter treats negative/out-of-range indices as unavailable SubJets.
        expressions, period_maps, sources = {}, {}, set()
        for period, cfg in self.configs.items():
            name, calculator = self._make_calculator(cfg)
            maps, available_sources = variation_maps(calculator)
            if None not in self.configs and any(
                "YEAR" in source for source in self.jes_unc
            ):
                # PR #121 deliberately exposes a common YEAR nuisance in a
                # combined campaign, evaluated with each period's own payload.
                year = str(cfg["year"])
                aliases = {
                    "jes" + source.replace("YEAR", year): "jes" + source
                    for source in self.jes_unc
                    if "YEAR" in source
                }

                def canonical(label):
                    for source, alias in aliases.items():
                        if label in (source + "up", source + "down"):
                            return alias + label[len(source) :]
                    return label

                maps = {
                    attr: {canonical(label): index for label, index in labels.items()}
                    for attr, labels in maps.items()
                }
                available_sources = [
                    aliases.get(source, source) for source in available_sources
                ]
            period_maps[period] = maps
            sources.update(available_sources)
            expressions[period] = f"{name}.produce({', '.join(args)})"
        if self.jer_variation_name != "jer" and self.jer_variation_name in sources:
            raise ValueError(
                "AK8 JER variation name collides with another backend source"
            )
        df = df.Define(
            "_fatjet_variations",
            period_switch(self.configs, expressions),
            excludeVariations=["*"],
        )

        def index_for(prop, source, tag):
            # Different periods may expose different YEAR-dependent JES sources or
            # mass-only indices. Missing sources use that period's nominal result.
            return period_switch(
                self.configs,
                {
                    period: str(maps[prop].get(source + tag, 0))
                    for period, maps in period_maps.items()
                },
            )

        nominal = {}
        for prop in ("pt", "mass", "msoftdrop"):
            nominal[prop] = (
                f"_fatjet_variations.{prop}(0)"
                if self.store_nominal
                else f"FatJet_{prop}"
            )
        df = df.Define(
            "_fatjet_order",
            "ROOT::VecOps::Reverse(ROOT::VecOps::Argsort(" + nominal["pt"] + "))",
            excludeVariations=["*"],
        )
        for prop in ("pt", "mass", "msoftdrop", "eta", "phi"):
            expression = nominal.get(prop, f"FatJet_{prop}")
            df = df.Define(
                f"CorrectedFatJet_{prop}",
                f"Take({expression}, _fatjet_order)",
                excludeVariations=["*"],
            )
        df = df.Define(
            "CorrectedFatJet_jetIdx",
            "ROOT::RVecI(_fatjet_order.begin(), _fatjet_order.end())",
            excludeVariations=["*"],
        )
        if self.store_variations:
            for source in sorted(sources):
                orders = []
                for tag in ("up", "down"):
                    index = index_for("pt", source, tag)
                    order = unique_name("_fatjet_order_")
                    df = df.Define(
                        order,
                        f"ROOT::VecOps::Reverse(ROOT::VecOps::Argsort(_fatjet_variations.pt({index})))",
                        excludeVariations=["*"],
                    )
                    orders.append(order)
                # Vary the index as well as all kinematics; downstream Take then propagates
                # tau variables, ID, taggers and subjet indices under reordered variations.
                for prop in ("pt", "mass", "msoftdrop", "eta", "phi", "jetIdx"):
                    expressions = []
                    for tag, order in zip(("up", "down"), orders):
                        if prop in ("pt", "mass", "msoftdrop"):
                            index = index_for(prop, source, tag)
                            expressions.append(
                                f"Take(_fatjet_variations.{prop}({index}), {order})"
                            )
                        elif prop == "jetIdx":
                            expressions.append(
                                f"ROOT::RVecI({order}.begin(), {order}.end())"
                            )
                        else:
                            expressions.append(f"Take(FatJet_{prop}, {order})")
                    rvec = "ROOT::RVecI" if prop == "jetIdx" else "ROOT::RVecF"
                    # Shared names vary both radii; an explicit AK8 JER name
                    # allows the analysis to use a separate resolution nuisance.
                    variation = self.jer_variation_name if source == "jer" else source
                    df = df.Vary(
                        f"CorrectedFatJet_{prop}",
                        "ROOT::RVec<" + rvec + ">{" + ", ".join(expressions) + "}",
                        ["up", "do"],
                        variation,
                    )
        df = df.Define("nCorrectedFatJet", "int(CorrectedFatJet_pt.size())")
        return df.DropColumns("_fatjet_*")

