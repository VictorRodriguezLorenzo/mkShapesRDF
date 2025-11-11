import ROOT
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Optional


def format_cutflow(label, passed_proxy, total_proxy):
    """Return a human-readable summary for a single cutflow entry."""

    total = total_proxy.GetValue()
    passed = passed_proxy.GetValue()

    if total > 0:
        efficiency = passed / total * 100.0
    else:
        efficiency = 0.0

    return [label, f"{passed} / {total} ({efficiency:.3f}%)"]

def flat_to_matrix_3x3(flat):
    """Convert flattened list/array of length 9 into a 3×3 matrix."""

    arr = np.array(flat, dtype=float)
    if arr.size != 9:
        raise ValueError("Expected 9 elements to build a 3x3 matrix")

    return arr.reshape((3, 3), order="F")


def build_h_perp_from_h(h_matrix):
    """Construct the H_perp matrix from the full H matrix."""

    h_matrix = np.asarray(h_matrix, dtype=float)
    if h_matrix.shape != (3, 3):
        raise ValueError("H matrix must be 3x3")

    h_perp = np.zeros((3, 3), dtype=float)
    h_perp[:2, :] = h_matrix[:2, :]
    h_perp[2, 2] = 1.0
    return h_perp

def plot_event(
    nu1_px,
    nu1_py,
    nu2_px,
    nu2_py,
    met_x,
    met_y,
    l1_pt_x,
    l1_pt_y,
    l2_pt_x,
    l2_pt_y,
    b1_pt_x,
    b1_pt_y,
    b2_pt_x,
    b2_pt_y,
    H1_flat,
    H2_flat,
    N1_flat,
    N2_flat,
    event_idx,
    output_dir: Optional[str] = None,
):

    if output_dir is None:
        output_dir = os.environ.get("NU_SOLUTION_PLOT_DIR")
    if not output_dir:
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    def arrow(x, y, label, color, lw=1.5):
        ax.arrow(
            0,
            0,
            x,
            y,
            head_width=2,
            length_includes_head=True,
            color=color,
            alpha=0.8,
            linewidth=lw,
        )
        ax.plot([], [], color=color, label=label)

    # Momentum vectors
    arrow(nu1_px, nu1_py, "Neutrino 1", "blue")
    arrow(nu2_px, nu2_py, "Neutrino 2", "pink")
    arrow(met_x, met_y, "MET", "red", lw=2)
    arrow(l1_pt_x, l1_pt_y, "Lepton 1", "lightblue")
    arrow(l2_pt_x, l2_pt_y, "Lepton 2", "purple")
    arrow(b1_pt_x, b1_pt_y, "B-jet 1", "cyan")
    arrow(b2_pt_x, b2_pt_y, "B-jet 2", "magenta")

    extents_x = [abs(nu1_px), abs(nu2_px), abs(met_x), abs(l1_pt_x), abs(l2_pt_x), abs(b1_pt_x), abs(b2_pt_x)]
    extents_y = [abs(nu1_py), abs(nu2_py), abs(met_y), abs(l1_pt_y), abs(l2_pt_y), abs(b1_pt_y), abs(b2_pt_y)]

    try:
        H1 = flat_to_matrix_3x3(H1_flat)
        H2 = flat_to_matrix_3x3(H2_flat)
    except ValueError:
        H1 = None
        H2 = None

    def draw_parametric_ellipse(h_matrix, label, color, transform=None):
        if h_matrix is None:
            return

        h_perp = build_h_perp_from_h(h_matrix)
        if not np.all(np.isfinite(h_perp)):
            return

        thetas = np.linspace(0.0, 2.0 * math.pi, num=361)
        points = []
        for theta in thetas:
            vec = np.array([math.cos(theta), math.sin(theta), 1.0])
            xy = h_perp.dot(vec)[:2]
            if transform is not None:
                xy = transform(xy)
            if not np.all(np.isfinite(xy)):
                continue
            points.append(xy)

        if not points:
            return

        pts = np.array(points)
        ax.plot(pts[:, 0], pts[:, 1], color=color, alpha=0.8, label=label)
        extents_x.extend(np.abs(pts[:, 0]))
        extents_y.extend(np.abs(pts[:, 1]))

    draw_parametric_ellipse(H1, "N₁ ellipse", "navy")
    draw_parametric_ellipse(
        H2,
        "N₂ ellipse",
        "darkorange",
        transform=lambda xy: np.array([met_x - xy[0], met_y - xy[1]]),
    )

    # Formatting
    ax.set_xlabel("pT_x (GeV)")
    ax.set_ylabel("pT_y (GeV)")
    ax.set_title(f"Event {event_idx}: Neutrino Solutions and Conic Ellipses")
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.axvline(0, color="black", lw=0.5, ls="--")

    if extents_x and extents_y:
        limit = max(max(extents_x), max(extents_y))
        if math.isfinite(limit) and limit > 0:
            ax.set_xlim(-1.2 * limit, 1.2 * limit)
            ax.set_ylim(-1.2 * limit, 1.2 * limit)

    ax.legend()
    ax.grid(True)
    ax.set_aspect("equal")

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(f"{output_dir}/event_{event_idx}_neutrino_solutions.png")
    plt.close(fig)


import ROOT
from mkShapesRDF.processor.framework.module import Module

class NuSolutionProducer(Module):
    def __init__(self):
        super().__init__("NuSolutionProducer")

    def runModule(self, df, values):
        ROOT.gInterpreter.Declare("""
        #include <Math/VectorUtil.h>
        #include <TLorentzVector.h>
        #include <TMatrixD.h>
        #include <TVectorD.h>
        #include <TVector3.h>
        #include <cmath>
        #include <vector>
        #include <array>
        #include <algorithm>
        #include <limits>
        #include <iostream>
        #include <memory>
        #include <functional>
        #include <string>
        #include <cstdlib>
        #include <TMatrixDSymEigen.h>
        #include <sstream>
        #include <iomanip>
                                  

        namespace nuana {

        // ---------- Utilities ----------

        bool debug_enabled() {
            static const bool enabled = [](){
                const char* env = std::getenv("NU_DEBUG");
                return env && std::string(env) != "0";
            }();
            return enabled;
        }

        void debug_log(const std::string& message) {
            if (debug_enabled()) {
                std::cout << "[nuana] " << message << std::endl;
            }
        }

        std::string format_tlv(const TLorentzVector& v) {
            std::ostringstream oss;
            oss.setf(std::ios::fixed);
            oss << std::setprecision(3)
                << "(px=" << v.Px()
                << ", py=" << v.Py()
                << ", pz=" << v.Pz()
                << ", E=" << v.E()
                << ", M=" << v.M() << ")";
            return oss.str();
        }

        std::string format_array2(const std::array<double,2>& arr) {
            std::ostringstream oss;
            oss.setf(std::ios::fixed);
            oss << std::setprecision(6)
                << "(" << arr[0] << ", " << arr[1] << ")";
            return oss.str();
        }

        std::string format_tvectord(const TVectorD& vec) {
            std::ostringstream oss;
            oss.setf(std::ios::fixed);
            oss << std::setprecision(6) << "[";
            for (int i = 0; i < vec.GetNrows(); ++i) {
                if (i != 0) oss << ", ";
                oss << vec(i);
            }
            oss << "]";
            return oss.str();
        }

        std::string format_matrix(const TMatrixD& mat) {
            std::ostringstream oss;
            oss.setf(std::ios::fixed);
            oss << std::setprecision(6);
            oss << "[";
            for (int r = 0; r < mat.GetNrows(); ++r) {
                if (r != 0) oss << ", ";
                oss << "[";
                for (int c = 0; c < mat.GetNcols(); ++c) {
                    if (c != 0) oss << ", ";
                    oss << mat(r, c);
                }
                oss << "]";
            }
            oss << "]";
            return oss.str();
        }

        // UnitCircle: returns a 3x3 matrix representing the unit circle in the F' coordinate system
        TMatrixD UnitCircle() {
            TMatrixD U(3,3);
            U.Zero();
            U(0,0) = 1.0;
            U(1,1) = 1.0;
            U(2,2) = -1.0;
            return U;
        }

        // Construct a robust 2x2 MET covariance matrix
        TMatrixD makeMetCov(double cov_xx, double cov_xy, double cov_yy) {
            TMatrixD sigma2(2, 2);
            sigma2(0, 0) = cov_xx;
            sigma2(0, 1) = cov_xy;
            sigma2(1, 0) = cov_xy;
            sigma2(1, 1) = cov_yy;

            bool finite = std::isfinite(cov_xx) && std::isfinite(cov_xy) && std::isfinite(cov_yy);
            double det = sigma2(0, 0) * sigma2(1, 1) - sigma2(0, 1) * sigma2(1, 0);
            double minDiag = std::min(sigma2(0, 0), sigma2(1, 1));

            if (!finite || minDiag <= 0.0 || det <= 0.0) {
                sigma2.UnitMatrix();
            }

            return sigma2;
        }

        double det3x3(const TMatrixD &M) {
            if (M.GetNrows() != 3 || M.GetNcols() != 3) {
                return 0.0;
            }
            const double a = M(0,0);
            const double b = M(0,1);
            const double c = M(0,2);
            const double d = M(1,0);
            const double e = M(1,1);
            const double f = M(1,2);
            const double g = M(2,0);
            const double h = M(2,1);
            const double i = M(2,2);
            return a * (e * i - f * h)
                 - b * (d * i - f * g)
                 + c * (d * h - e * g);
        }

        bool invert3x3(const TMatrixD &M, TMatrixD &inv, double tol = 1e-12) {
            if (debug_enabled()) {
                debug_log("invert3x3: attempting inversion with tol=" + std::to_string(tol) +
                          ", matrix=" + format_matrix(M));
            }

            double det = det3x3(M);
            if (!std::isfinite(det) || std::abs(det) <= tol) {
                if (debug_enabled()) {
                    debug_log("invert3x3: singular matrix encountered, det=" + std::to_string(det));
                }
                return false;
            }

            inv.ResizeTo(3, 3);

            inv(0,0) =  (M(1,1) * M(2,2) - M(1,2) * M(2,1)) / det;
            inv(0,1) = -(M(0,1) * M(2,2) - M(0,2) * M(2,1)) / det;
            inv(0,2) =  (M(0,1) * M(1,2) - M(0,2) * M(1,1)) / det;
            inv(1,0) = -(M(1,0) * M(2,2) - M(1,2) * M(2,0)) / det;
            inv(1,1) =  (M(0,0) * M(2,2) - M(0,2) * M(2,0)) / det;
            inv(1,2) = -(M(0,0) * M(1,2) - M(0,2) * M(1,0)) / det;
            inv(2,0) =  (M(1,0) * M(2,1) - M(1,1) * M(2,0)) / det;
            inv(2,1) = -(M(0,0) * M(2,1) - M(0,1) * M(2,0)) / det;
            inv(2,2) =  (M(0,0) * M(1,1) - M(0,1) * M(1,0)) / det;

            if (debug_enabled()) {
                debug_log("invert3x3: successful inversion, det=" + std::to_string(det) +
                          ", inverse=" + format_matrix(inv));
            }

            return true;
        }

        bool invert2x2(const TMatrixD &M, TMatrixD &inv, double tol = 1e-12) {
            if (M.GetNrows() != 2 || M.GetNcols() != 2) {
                return false;
            }

            double det = M(0,0) * M(1,1) - M(0,1) * M(1,0);
            if (!std::isfinite(det) || std::abs(det) <= tol) {
                return false;
            }

            inv.ResizeTo(2, 2);
            const double invDet = 1.0 / det;
            inv(0,0) =  M(1,1) * invDet;
            inv(0,1) = -M(0,1) * invDet;
            inv(1,0) = -M(1,0) * invDet;
            inv(1,1) =  M(0,0) * invDet;
            return true;
        }

        // Cofactor for 3x3 matrix A at (i,j)
        double cofactor(const TMatrixD &A, int i, int j) {
            // Compute determinant of minor (2x2) by skipping row i and column j
            int rows[2], cols[2], r = 0, c = 0;
            for (int idx = 0; idx < 3; ++idx) {
            if (idx != i) rows[r++] = idx;
            if (idx != j) cols[c++] = idx;
            }
            double det2 = A(rows[0], cols[0]) * A(rows[1], cols[1]) - A(rows[0], cols[1]) * A(rows[1], cols[0]);
            return ((i + j) % 2 == 0 ? 1.0 : -1.0) * det2;
        }

        TMatrixD Rotation(int axis, double angle) {
            const double c = std::cos(angle);
            const double s = std::sin(angle);

            TMatrixD R(3, 3);
            R.UnitMatrix();
            R *= c;

            for (int i = -1; i <= 1; ++i) {
                const int row = (axis - i + 3) % 3;
                const int col = (axis + i + 3) % 3;
                R(row, col) = i * s + (1 - i * i);
            }

            return R;
        }

        TMatrixD Derivative() {
            // Matrix to differentiate [cos(theta), sin(theta), 1]
            double angle = M_PI / 2.0;
            double c = std::cos(angle);
            double s = std::sin(angle);
            TMatrixD rot(3,3);
            rot.UnitMatrix();
            rot(0,0) = c; rot(0,1) = -s;
            rot(1,0) = s; rot(1,1) = c;
            rot(2,2) = 1.0;

            TMatrixD diag(3,3);
            diag.Zero();
            diag(0,0) = 1.0;
            diag(1,1) = 1.0;
            diag(2,2) = 0.0;

            TMatrixD result = rot * diag;
            return result;
        }

        // Valid real sqrt solutions of y = x^2
        std::vector<double> multisqrt(double y) {
            if (y < 0.0) {
                return {};
            } else if (y == 0.0) {
                return {0.0};
            } else {
                double r = std::sqrt(y);
                return {-r, r};
            }
        }

        // factor_degenerate: linear factors (lines) for degenerate quadratic (3x3 symmetric G)
        std::vector<std::array<double,3>> factor_degenerate(
            const TMatrixD &G, double zero = 0.0
        ) {
            std::vector<std::array<double,3>> lines;

            if (std::abs(G(0,0)) <= zero && std::abs(G(1,1)) <= zero) {
                lines.push_back({G(0,1), 0.0, G(1,2)});
                lines.push_back({0.0, G(0,1), G(0,2) - G(1,2)});
                return lines;
            }

            bool swapXY = std::abs(G(0,0)) > std::abs(G(1,1));
            TMatrixD Q(G);
            if (swapXY) {
                TMatrixD tmp(3,3);
                int order[3] = {1, 0, 2};
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 3; ++c)
                        tmp(r,c) = Q(order[r], order[c]);
                Q = tmp;
            }

            double denom = Q(1,1);
            if (std::abs(denom) <= zero) {
                return lines;
            }

            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    Q(r,c) /= denom;

            double q22 = cofactor(Q, 2, 2);

            auto swap_back = [](const std::array<double,3> &L) {
                return std::array<double,3>{L[1], L[0], L[2]};
            };

            if (-q22 <= zero) {
                double cof00 = cofactor(Q, 0, 0);
                auto roots = multisqrt(-cof00);
                for (double sVal : roots) {
                    std::array<double,3> L{Q(0,1), Q(1,1), Q(1,2) + sVal};
                    if (swapXY) {
                        L = swap_back(L);
                    }
                    lines.push_back(L);
                }
            } else {
                double x0 = cofactor(Q, 0, 2) / q22;
                double y0 = cofactor(Q, 1, 2) / q22;
                auto roots = multisqrt(-q22);
                for (double sVal : roots) {
                    double m = Q(0,1) + sVal;
                    std::array<double,3> L{m, Q(1,1), -Q(1,1) * y0 - m * x0};
                    if (swapXY) {
                        L = swap_back(L);
                    }
                    lines.push_back(L);
                }
            }

            return lines;
        }


        std::vector<TVectorD> intersections_ellipse_line(
            const TMatrixD &ellipse,
            const std::array<double,3> &line,
            double zero = 1e-12
        ) {
            TMatrixD cross(3,3);
            for (int i = 0; i < 3; ++i) {
                TVectorD row(3);
                for (int j = 0; j < 3; ++j) {
                    row[j] = ellipse(i,j);
                }
                cross(i,0) = line[1]*row[2] - line[2]*row[1];
                cross(i,1) = line[2]*row[0] - line[0]*row[2];
                cross(i,2) = line[0]*row[1] - line[1]*row[0];
            }

            TMatrixD crossT(TMatrixD::kTransposed, cross);
            TMatrixDEigen eig(crossT);
            const TMatrixD &eigVecs = eig.GetEigenVectors();

            std::vector<std::pair<TVectorD,double>> candidates;

            for (int i = 0; i < 3; ++i) {
                TVectorD v(3);
                for (int j = 0; j < 3; ++j) {
                    v[j] = eigVecs(j, i);
                }

                if (std::abs(v[2]) <= zero) {
                    continue;
                }

                double inv = 1.0 / v[2];
                TVectorD s_vec(3);
                s_vec[0] = v[0] * inv;
                s_vec[1] = v[1] * inv;
                s_vec[2] = 1.0;

                double lv = line[0]*v[0] + line[1]*v[1] + line[2]*v[2];
                TVectorD Ev = ellipse * v;
                double vev = v[0]*Ev[0] + v[1]*Ev[1] + v[2]*Ev[2];

                double k = lv*lv + vev*vev;
                candidates.emplace_back(s_vec, k);
            }

            std::sort(candidates.begin(), candidates.end(),
                      [](const auto &a, const auto &b) { return a.second < b.second; });

            if (candidates.size() > 2) {
                candidates.resize(2);
            }

            std::vector<TVectorD> result;
            for (const auto &entry : candidates) {
                if (entry.second < zero) {
                    result.push_back(entry.first);
                }
            }
            return result;
        }


        std::pair<std::vector<TVectorD>, std::vector<std::array<double,3>>>
        intersections_ellipses(
            const TMatrixD &A, const TMatrixD &B,
            bool returnLines = false, double zero = 1e-10
        ) {
            double detA = det3x3(A);
            double detB = det3x3(B);

            const TMatrixD *AA = &A;
            const TMatrixD *BB = &B;
            if (std::abs(detB) > std::abs(detA)) {
                AA = &B;
                BB = &A;
            }

            TMatrixD invA;
            if (!invert3x3(*AA, invA)) {
                return {std::vector<TVectorD>{}, std::vector<std::array<double,3>>{}};
            }

            TMatrixD M = invA * (*BB);
            TMatrixDEigen eig(M);
            TVectorD evalsRe = eig.GetEigenValuesRe();
            TVectorD evalsIm = eig.GetEigenValuesIm();

            double eigenvalue = 0.0;
            bool found = false;
            for (int i = 0; i < evalsRe.GetNrows(); ++i) {
                if (std::abs(evalsIm(i)) <= zero) {
                    eigenvalue = evalsRe(i);
                    found = true;
                    break;
                }
            }
            if (!found) {
                return {std::vector<TVectorD>{}, std::vector<std::array<double,3>>{}};
            }

            TMatrixD G = (*BB) - eigenvalue * (*AA);
            auto lines = factor_degenerate(G, zero);

            std::vector<TVectorD> points;
            for (const auto &line : lines) {
                auto pts = intersections_ellipse_line(*AA, line, zero);
                points.insert(points.end(), pts.begin(), pts.end());
            }

            if (returnLines) {
                return {points, lines};
            }
            return {points, std::vector<std::array<double,3>>{}};
        }


        struct nuSolutionSet {
            TLorentzVector b, mu;
            double c, s, x0, x0p, Sx, Sy, w, w_, x1, y1, Z, Om2, eps2, mW2;
            mutable bool usedMatrixFallback_;

            nuSolutionSet()
                : b(), mu(), c(1.0), s(0.0), x0(0.0), x0p(0.0), Sx(0.0), Sy(0.0), w(0.0), w_(0.0),
                  x1(0.0), y1(0.0), Z(0.0), Om2(1.0), eps2(0.0), mW2(80.385 * 80.385),
                  usedMatrixFallback_(false)
            {
                b.SetPxPyPzE(0.0, 0.0, 0.0, 1.0);
                mu.SetPxPyPzE(0.0, 0.0, 0.0, 1.0);
            }

            nuSolutionSet(const TLorentzVector& b_, const TLorentzVector& mu_,
                  double mW = 80.385, double mT = 172.5, double mN = 0.0)
                : b(b_), mu(mu_), usedMatrixFallback_(false)
            {
                mW2 = mW * mW;
                double mT2 = mT * mT;
                double mN2 = mN * mN;

            if (debug_enabled()) {
                std::ostringstream oss;
                oss << "nuSolutionSet: constructing with b=" << format_tlv(b)
                    << ", mu=" << format_tlv(mu)
                    << ", mW=" << mW
                    << ", mT=" << mT
                    << ", mN=" << mN;
                debug_log(oss.str());
            }

            c = ROOT::Math::VectorUtil::CosTheta(b, mu);
            s = std::sqrt(std::max(0.0, 1.0 - c * c));
            if (s < 1e-12) {
                s = 1e-12;
            }

            if (debug_enabled()) {
                debug_log("nuSolutionSet: cos(theta)=" + std::to_string(c) +
                          ", sin(theta)=" + std::to_string(s));
            }

            x0p = - (mT2 - mW2 - b.M2()) / (2.0 * b.E());
            x0  = - (mW2 - mu.M2() - mN2) / (2.0 * mu.E());

            double Bb = b.Beta();
            double Bm = mu.Beta();
            if (std::abs(Bb) < 1e-12) {
                Bb = (Bb >= 0 ? 1e-12 : -1e-12);
            }

            if (debug_enabled()) {
                std::ostringstream oss;
                oss << "nuSolutionSet: x0=" << x0 << ", x0p=" << x0p
                    << ", Bb=" << Bb << ", Bm=" << Bm;
                debug_log(oss.str());
            }

            const double Bm2 = std::max(1e-12, Bm * Bm);
            Sx = (x0 * Bm - mu.P() * (1.0 - Bm * Bm)) / Bm2;
            Sy = (x0p / Bb - c * Sx) / s;

            w  = (Bm / Bb - c) / s;
            w_ = (-Bm / Bb - c) / s;

            if (debug_enabled()) {
                std::ostringstream oss;
                oss << "nuSolutionSet: w=" << w << ", w_=" << w_;
                debug_log(oss.str());
            }

            Om2 = w * w + 1.0 - Bm * Bm;
            eps2 = (mW2 - mN2) * (1.0 - Bm * Bm);

            x1 = Sx - (Sx + w * Sy) / Om2;
            y1 = Sy - (Sx + w * Sy) * w / Om2;

            double Z2 = x1 * x1 * Om2 - (Sy - w * Sx) * (Sy - w * Sx) - (mW2 - x0 * x0 - eps2);
            if (debug_enabled()) {
                debug_log("nuSolutionSet: computed parameters Sx=" + std::to_string(Sx) +
                          ", Sy=" + std::to_string(Sy) +
                          ", w=" + std::to_string(w) +
                          ", Om2=" + std::to_string(Om2) +
                          ", eps2=" + std::to_string(eps2) +
                          ", Z2=" + std::to_string(Z2));
            }
            //cout << "Z2: " << Z2 << std::endl;
            //cout << "Om2: " << Om2 << std::endl;  
            //cout << "eps2: " << eps2 << std::endl;
            //cout << "x0: " << x0 << ", x0p: " << x0p << std::endl;
            //cout << "Sx: " << Sx << ", Sy: " << Sy << std::endl;
            //cout << "w: " << w << ", w_: " << w_ << std::endl;
            //cout << "x1: " << x1 << ", y1: " << y1 << std::endl;
            //cout << "c: " << c << ", s: " << s << std::endl;
            // Calculate Z
            Z = std::sqrt(std::max(0.0, Z2));
            if (debug_enabled() && Z2 < 0.0) {
                debug_log("nuSolutionSet: Z2 negative, clamped to zero before sqrt.");
            }
            }

            // Extended rotation from F' to F coord.
            TMatrixD getK() const {
                TMatrixD K(4,4);
                K(0,0) = c; K(0,1) = -s; K(0,2) = 0; K(0,3) = 0;
                K(1,0) = s; K(1,1) =  c; K(1,2) = 0; K(1,3) = 0;
                K(2,0) = 0; K(2,1) =  0; K(2,2) = 1; K(2,3) = 0;
                K(3,0) = 0; K(3,1) =  0; K(3,2) = 0; K(3,3) = 1;
                return K;
            }

            // F coord. constraint on W momentum: ellipsoid.
            TMatrixD getA_mu_mat() const {
                TMatrixD A(4,4);
                const double B2 = mu.Beta() * mu.Beta();
                const double SxB2 = Sx * B2;
                double F = mW2 - x0 * x0 - eps2;

                A(0,0) = 1 - B2; A(0,1) = 0;    A(0,2) = 0; A(0,3) = SxB2;
                A(1,0) = 0;      A(1,1) = 1;    A(1,2) = 0; A(1,3) = 0;
                A(2,0) = 0;      A(2,1) = 0;    A(2,2) = 1; A(2,3) = 0;
                A(3,0) = SxB2;   A(3,1) = 0;    A(3,2) = 0; A(3,3) = F;

                return A;
            }

                                  
            // F coord. constraint on W momentum: ellipsoid
            TMatrixD A_b() const {
                const TMatrixD K = getK();
                const double B = b.Beta();

                TMatrixD A_b_(4, 4);
                A_b_(0,0) = 1 - B*B;  A_b_(0,1) = 0;      A_b_(0,2) = 0;      A_b_(0,3) = B*x0p;
                A_b_(1,0) = 0;        A_b_(1,1) = 1;      A_b_(1,2) = 0;      A_b_(1,3) = 0;
                A_b_(2,0) = 0;        A_b_(2,1) = 0;      A_b_(2,2) = 1;      A_b_(2,3) = 0;
                A_b_(3,0) = B*x0p;    A_b_(3,1) = 0;      A_b_(3,2) = 0;      A_b_(3,3) = mW2 - x0p*x0p;

                TMatrixD result = K * A_b_;
                TMatrixD KT(TMatrixD::kTransposed, K);
                result *= KT;
                return result;
            }
            
                                  
            // Rotation from F coord. to laboratory coord.
            TMatrixD getR_T() const {
                auto apply = [](const TMatrixD& R, const TVector3& vec) {
                    TVector3 out;
                    for (int i = 0; i < 3; ++i) {
                        out[i] = R(i,0) * vec.X() + R(i,1) * vec.Y() + R(i,2) * vec.Z();
                    }
                    return out;
                };

                const TMatrixD Rz = Rotation(2, -mu.Phi());
                const TMatrixD Ry = Rotation(1, 0.5 * M_PI - mu.Theta());
                TVector3 rotated = apply(Ry, apply(Rz, b.Vect()));
                const TMatrixD Rx = Rotation(0, -std::atan2(rotated.Z(), rotated.Y()));

                TMatrixD RzT(TMatrixD::kTransposed, Rz);
                TMatrixD RyT(TMatrixD::kTransposed, Ry);
                TMatrixD RxT(TMatrixD::kTransposed, Rx);

                TMatrixD tmp = RyT * RxT;
                return RzT * tmp;
            }

            // H_tilde transformation from t=[c,s,1] to p_nu: F coord.
            TMatrixD getH_tilde() const {
                double _x1 = x1;
                double _y1 = y1;
                double p = mu.P();
                double _Z = Z;
                double _w = w;
                double _Om = std::sqrt(Om2);
                TMatrixD H_tilde(3, 3);
                //cout << "Om: " << _Om << std::endl;
                //cout << "Z: " << _Z << std::endl;
                //cout << "x1: " << _x1 << ", y1: " << _y1 << ", p: " << p << std::endl;
                H_tilde(0,0) =  _Z/_Om; H_tilde(0,1) = 0; H_tilde(0,2) = _x1-p;
                H_tilde(1,0) = _w*_Z/_Om; H_tilde(1,1) = 0; H_tilde(1,2) = _y1;
                H_tilde(2,0) = 0; H_tilde(2,1) = _Z; H_tilde(2,2) = 0;
                return H_tilde;
            }

            // Transformation of t=[c,s,1] to p_nu: lab coord.
            TMatrixD getH() const {
                TMatrixD result = getR_T() * getH_tilde();
                if (debug_enabled()) {
                    debug_log("nuSolutionSet::getH: result=" + format_matrix(result));
                }
                return result;
            }

            // Transformation of t=[c,s,1] to pT_nu: lab coord.
            TMatrixD getH_perp() const {
                TMatrixD h = getH();
                TMatrixD h_perp(3,3);
                h_perp.Zero();
                for (int i = 0; i < 2; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        h_perp(i,j) = h(i,j);
                    }
                }
                h_perp(2,2) = 1.0;
                if (debug_enabled()) {
                    debug_log("nuSolutionSet::getH_perp: result=" + format_matrix(h_perp));
                }
                return h_perp;
            }


            // Solution ellipse of pT_nu: lab coord.
            TMatrixD getN() const {
                // Invert H_perp safely, falling back to identity if singular
                TMatrixD Hp = getH_perp();
                TMatrixD HpInv(Hp);
                usedMatrixFallback_ = false;

                const double det = det3x3(Hp);
                if (!std::isfinite(det) || std::abs(det) < 1e-12) {
                    usedMatrixFallback_ = true;
                    if (debug_enabled()) {
                        debug_log("nuSolutionSet::getN: inversion failed, using identity.");
                    }
                    TMatrixD fallback(3,3);
                    fallback.UnitMatrix();
                    return fallback;
                }

                HpInv.Invert();
                TMatrixD HpInvT(TMatrixD::kTransposed, HpInv);
                TMatrixD result = HpInvT * UnitCircle() * HpInv;
                if (debug_enabled()) {
                    debug_log("nuSolutionSet::getN: result=" + format_matrix(result));
                }
                return result;
            }

            bool usedMatrixFallback() const { return usedMatrixFallback_; }
        };

        // ---------- Classes ----------
        // singleNeutrinoSolution: finds the best single-neutrino solution for given b-jet and lepton momenta.
        class singleNeutrinoSolution {
        public:
            nuana::nuSolutionSet solutionSet;
            TMatrixD X;
            std::vector<TVectorD> solutions;

            singleNeutrinoSolution()
                : solutionSet(), X(3, 3), solutions()
            {
                X.Zero();
            }

            singleNeutrinoSolution(const TLorentzVector& b, const TLorentzVector& mu,
                                  double metX, double metY,
                                  const TMatrixD& sigma2,
                                  double mW2 = 80.385*80.385, double mT2 = 172.5*172.5)
                : solutionSet(b, mu, std::sqrt(mW2), std::sqrt(mT2))
            {
                if (debug_enabled()) {
                    std::ostringstream oss;
                    oss << "singleNeutrinoSolution: inputs b=" << format_tlv(b)
                        << ", mu=" << format_tlv(mu)
                        << ", metX=" << metX
                        << ", metY=" << metY;
                    debug_log(oss.str());
                }
                // Build S2: inverse of sigma2, padded to 3x3
                TMatrixD S2(3,3); S2.Zero();
                TMatrixD sigma2_inv;
                if (!invert2x2(sigma2, sigma2_inv)) {
                    debug_log("singleNeutrinoSolution: MET covariance inversion failed, using identity.");
                    sigma2_inv.ResizeTo(2,2);
                    sigma2_inv.UnitMatrix();
                }
                for (int i=0;i<2;++i) for (int j=0;j<2;++j)
                    S2(i,j) = sigma2_inv(i,j);

                // V0: outer([metX, metY, 0], [0,0,1])
                TMatrixD V0(3,3); V0.Zero();
                V0(0,2) = metX;
                V0(1,2) = metY;
                V0(2,2) = 0.0;

                // deltaNu = V0 - H
                TMatrixD deltaNu = V0;
                TMatrixD H = solutionSet.getH();
                for (int i=0;i<3;++i) for (int j=0;j<3;++j)
                    deltaNu(i,j) -= H(i,j);

                // X = deltaNu^T * S2 * deltaNu
                TMatrixD deltaNu_T(TMatrixD::kTransposed, deltaNu);
                X = deltaNu_T * S2 * deltaNu;

                if (debug_enabled()) {
                    debug_log("singleNeutrinoSolution: constraint matrix X=" + format_matrix(X));
                }

                // M = X * Derivative() + (X * Derivative()).T
                TMatrixD XD = X * Derivative();
                TMatrixD M = XD;
                TMatrixD XD_T(TMatrixD::kTransposed, XD);
                for (int i=0;i<3;++i) for (int j=0;j<3;++j)
                    M(i,j) += XD_T(i,j);

                // Find intersections
                solutions = intersections_ellipses(M, UnitCircle()).first;

                if (debug_enabled()) {
                    debug_log("singleNeutrinoSolution: found " + std::to_string(solutions.size()) + " candidate solutions");
                    for (size_t idx = 0; idx < solutions.size(); ++idx) {
                        debug_log("singleNeutrinoSolution: solution[" + std::to_string(idx) + "]=" + format_tvectord(solutions[idx]));
                    }
                }


                // Sort solutions by chi2
                std::sort(solutions.begin(), solutions.end(),
                    [this](const TVectorD& a, const TVectorD& b) {
                        return calcX2(a) < calcX2(b);
                    });
            }

            double calcX2(const TVectorD& t) const {
                TVectorD Xt = X * t;
                double val = 0.0;
                for (int i=0;i<3;++i) val += t(i) * Xt(i);
                return val;
            }

            double chi2() const {
                if (solutions.empty()) return invalidValue();
                double chi2_val = calcX2(solutions[0]);
                if (!std::isfinite(chi2_val)) return invalidValue();
                if (debug_enabled()) {
                    debug_log("singleNeutrinoSolution: best chi2=" + std::to_string(chi2_val));
                }
                return chi2_val;
            }

            bool hasSolutions() const {
                return !solutions.empty();
            }

            bool isValid() const {
                const auto& nu_vec = bestNu();
                return std::isfinite(nu_vec(0)) && std::isfinite(nu_vec(1)) && std::isfinite(nu_vec(2));
            }

            TVectorD nu() const {
                return bestNu();
            }

            double nu_px() const { return bestNu()(0); }
            double nu_py() const { return bestNu()(1); }
            double nu_pz() const { return bestNu()(2); }

            double nu_pt() const {
                const auto& nu_vec = bestNu();
                if (!std::isfinite(nu_vec(0)) || !std::isfinite(nu_vec(1))) return invalidValue();
                return std::hypot(nu_vec(0), nu_vec(1));
            }

            double nu_phi() const {
                const auto& nu_vec = bestNu();
                if (!std::isfinite(nu_vec(0)) || !std::isfinite(nu_vec(1))) return invalidValue();
                return std::atan2(nu_vec(1), nu_vec(0));
            }

            double nu_energy() const {
                const auto& nu_vec = bestNu();
                if (!std::isfinite(nu_vec(0)) || !std::isfinite(nu_vec(1)) || !std::isfinite(nu_vec(2))) return invalidValue();
                double pt = std::hypot(nu_vec(0), nu_vec(1));
                return std::sqrt(pt * pt + nu_vec(2) * nu_vec(2));
            }

            bool usedMatrixFallback() const {
                return solutionSet.usedMatrixFallback();
            }

        private:
            mutable TVectorD cachedNu_;
            mutable bool cachedReady_ = false;

            static double invalidValue() {
                return std::numeric_limits<double>::quiet_NaN();
            }

            void fillInvalid() const {
                cachedNu_.ResizeTo(3);
                double nan = invalidValue();
                for (int i = 0; i < 3; ++i) {
                    cachedNu_(i) = nan;
                }
            }

            const TVectorD& bestNu() const {
                if (!cachedReady_) {
                    if (solutions.empty()) {
                        fillInvalid();
                    } else {
                        cachedNu_ = solutionSet.getH() * solutions[0];
                        bool finite = true;
                        for (int i = 0; i < 3; ++i) {
                            if (!std::isfinite(cachedNu_(i))) {
                                finite = false;
                                break;
                            }
                        }
                        if (!finite) {
                            fillInvalid();
                        }
                    }
                    cachedReady_ = true;
                }
                return cachedNu_;
            }
        };
                                  

        // doubleNeutrinoSolution: finds the best double-neutrino solution for given b-jet and lepton momenta.
        class doubleNeutrinoSolution {
        public:
            struct NuPair {
                std::array<double, 2> first;
                std::array<double, 2> second;
            };

            doubleNeutrinoSolution()
                : H1(3, 3), H2(3, 3), N1_(3, 3), N2_(3, 3), usedMinimizerFallback_(false)
            {
                H1.Zero();
                H2.Zero();
                N1_.Zero();
                N2_.Zero();
            }

            doubleNeutrinoSolution(
                const TLorentzVector& b1,
                const TLorentzVector& b2,
                const TLorentzVector& l1,
                const TLorentzVector& l2,
                double met_x,
                double met_y)
            {
                const double mW = 80.385; // W mass in GeV
                const double mT = 172.5;  // Top mass in GeV

                auto try_pairing = [&](const TLorentzVector& B1, const TLorentzVector& B2,
                                       const TLorentzVector& L1, const TLorentzVector& L2) {
                    PairingResult result;

                    if (debug_enabled()) {
                        std::ostringstream oss;
                        oss << "doubleNeutrinoSolution::try_pairing: B1=" << format_tlv(B1)
                            << ", B2=" << format_tlv(B2)
                            << ", L1=" << format_tlv(L1)
                            << ", L2=" << format_tlv(L2);
                        debug_log(oss.str());
                    }

                    nuana::nuSolutionSet ss1(B1, L1, mW, mT);
                    nuana::nuSolutionSet ss2(B2, L2, mW, mT);

                    TMatrixD H1tmp = ss1.getH();
                    TMatrixD H2tmp = ss2.getH();
                    result.H1.ResizeTo(H1tmp.GetNrows(), H1tmp.GetNcols());
                    result.H2.ResizeTo(H2tmp.GetNrows(), H2tmp.GetNcols());
                    result.H1 = H1tmp;
                    result.H2 = H2tmp;

                    TMatrixD V0(3, 3);
                    V0.Zero();
                    V0(0, 2) = met_x;
                    V0(1, 2) = met_y;
                    V0(2, 2) = 0.0;

                    TMatrixD S = V0 - UnitCircle();

                    TMatrixD N1 = ss1.getN();
                    TMatrixD N2 = ss2.getN();

                    TMatrixD n2 = S.T() * N2 * S;

                    result.N1.ResizeTo(N1.GetNrows(), N1.GetNcols());
                    result.N2.ResizeTo(n2.GetNrows(), n2.GetNcols());
                    result.N1 = N1;
                    result.N2 = n2;

                    std::vector<TVectorD> intersections =
                        nuana::intersections_ellipses(N1, n2).first;

                    if (debug_enabled()) {
                        debug_log("doubleNeutrinoSolution::try_pairing: intersections=" + std::to_string(intersections.size()));
                    }

                    for (const auto& sol : intersections) {
                        TVectorD nu1 = ss1.getH() * sol;
                        TVectorD nu2 = S * sol;

                        NuPair pair;
                        pair.first = {nu1(0), nu1(1)};
                        pair.second = {nu2(0), nu2(1)};

                        if (isFinitePair(pair)) {
                            result.solutions.push_back(pair);
                            if (debug_enabled()) {
                                debug_log("doubleNeutrinoSolution::try_pairing: added intersection solution nu1=" +
                                          format_array2(pair.first) + ", nu2=" + format_array2(pair.second));
                            }
                        }
                    }

                    if (result.solutions.empty()) {
                        if (debug_enabled()) {
                            debug_log("doubleNeutrinoSolution::try_pairing: no direct intersections, invoking minimizer fallback.");
                        }
                        TMatrixD es1 = ss1.getH_perp();
                        TMatrixD es2 = ss2.getH_perp();

                        TVectorD met_vec(3);
                        met_vec(0) = met_x;
                        met_vec(1) = met_y;
                        met_vec(2) = 1.0;

                        auto nus = [&](const TVectorD& ts) {
                            std::vector<TVectorD> momenta;
                            momenta.reserve(2);
                            for (int i = 0; i < 2; ++i) {
                                TVectorD vec(3);
                                vec(0) = std::cos(ts(i));
                                vec(1) = std::sin(ts(i));
                                vec(2) = 1.0;

                                TVectorD nu = (i == 0) ? es1 * vec : es2 * vec;
                                momenta.push_back(nu);
                            }
                            return momenta;
                        };

                        auto residuals = [&](const TVectorD& params) {
                            auto nu_vecs = nus(params);
                            TVectorD total = nu_vecs[0] + nu_vecs[1] - met_vec;
                            TVectorD res(2);
                            res(0) = total(0);
                            res(1) = total(1);
                            return res;
                        };

                        class ResidualsFunction : public ROOT::Math::IMultiGenFunction {
                        public:
                            explicit ResidualsFunction(
                                const std::function<TVectorD(const TVectorD&)>& f)
                                : func(f) {}

                            unsigned int NDim() const override { return 2; }

                            double DoEval(const double* x) const override {
                                TVectorD params(2);
                                params(0) = x[0];
                                params(1) = x[1];
                                TVectorD res = func(params);
                                return res(0) * res(0) + res(1) * res(1);
                            }

                            ROOT::Math::IMultiGenFunction* Clone() const override {
                                return new ResidualsFunction(func);
                            }

                        private:
                            std::function<TVectorD(const TVectorD&)> func;
                        };

                        std::unique_ptr<ROOT::Math::Minimizer> min(
                            ROOT::Math::Factory::CreateMinimizer("Minuit2", "Migrad"));

                        if (min) {
                            min->SetTolerance(1e-10);
                            min->SetPrecision(1e-12);
                            min->SetVariableStepSize(0, 0.01);
                            min->SetVariableStepSize(1, 0.01);

                            ResidualsFunction residualsFunc(residuals);
                            min->SetFunction(residualsFunc);

                            min->SetVariable(0, "t1", 0.0, 0.1);
                            min->SetVariable(1, "t2", 0.0, 0.1);

                            if (min->Minimize()) {
                                TVectorD ts(2);
                                ts(0) = min->X()[0];
                                ts(1) = min->X()[1];

                                auto fallbackSolutions = nus(ts);

                                NuPair pair;
                                pair.first = {fallbackSolutions[0](0), fallbackSolutions[0](1)};
                                pair.second = {fallbackSolutions[1](0), fallbackSolutions[1](1)};
                                result.solutions.push_back(pair);
                                result.usedMinimizerFallback = true;
                                if (debug_enabled()) {
                                    debug_log("doubleNeutrinoSolution::try_pairing: minimizer produced solution nu1=" +
                                              format_array2(pair.first) + ", nu2=" + format_array2(pair.second));
                                }
                            } else {
                                debug_log("doubleNeutrinoSolution::try_pairing: minimizer failed to converge.");
                            }
                        } else {
                            debug_log("doubleNeutrinoSolution::try_pairing: failed to create Minuit2 minimizer instance.");
                        }
                    }

                    if (debug_enabled()) {
                        debug_log("doubleNeutrinoSolution::try_pairing: returning " +
                                  std::to_string(result.solutions.size()) + " solutions" +
                                  (result.usedMinimizerFallback ? " (fallback used)" : ""));
                    }

                    return result;
                };

                PairingResult pairing1 = try_pairing(b1, b2, l1, l2);
                PairingResult pairing2 = try_pairing(b1, b2, l2, l1);

                double residual1 = metResidual(pairing1, met_x, met_y);
                double residual2 = metResidual(pairing2, met_x, met_y);
                if (debug_enabled()) {
                    debug_log("doubleNeutrinoSolution: residual pairing1=" + std::to_string(residual1) +
                              ", pairing2=" + std::to_string(residual2));
                }

                if (residual1 <= residual2) {
                    nunu_s = pairing1.solutions;
                    H1.ResizeTo(pairing1.H1.GetNrows(), pairing1.H1.GetNcols());
                    H2.ResizeTo(pairing1.H2.GetNrows(), pairing1.H2.GetNcols());
                    H1 = pairing1.H1;
                    H2 = pairing1.H2;
                    N1_ = pairing1.N1;
                    N2_ = pairing1.N2;
                    usedMinimizerFallback_ = pairing1.usedMinimizerFallback;
                    if (debug_enabled()) {
                        debug_log("doubleNeutrinoSolution: selected pairing1 with " +
                                  std::to_string(nunu_s.size()) + " solutions");
                    }
                } else {
                    nunu_s = pairing2.solutions;
                    H1.ResizeTo(pairing2.H1.GetNrows(), pairing2.H1.GetNcols());
                    H2.ResizeTo(pairing2.H2.GetNrows(), pairing2.H2.GetNcols());
                    H1 = pairing2.H1;
                    H2 = pairing2.H2;
                    N1_ = pairing2.N1;
                    N2_ = pairing2.N2;
                    usedMinimizerFallback_ = pairing2.usedMinimizerFallback;
                    if (debug_enabled()) {
                        debug_log("doubleNeutrinoSolution: selected pairing2 with " +
                                  std::to_string(nunu_s.size()) + " solutions");
                    }
                }

                if (nunu_s.empty()) {
                    NuPair empty_pair{{std::numeric_limits<double>::quiet_NaN(),
                                       std::numeric_limits<double>::quiet_NaN()},
                                      {std::numeric_limits<double>::quiet_NaN(),
                                       std::numeric_limits<double>::quiet_NaN()}};
                    nunu_s.push_back(empty_pair);
                    debug_log("doubleNeutrinoSolution: no solutions found, inserting NaN placeholders.");
                }
            }

            std::vector<NuPair> get_nunu_s() const {
                return nunu_s;
            }

            const TMatrixD& getH1() const { return H1; }
            const TMatrixD& getH2() const { return H2; }
            const TMatrixD& getN1() const { return N1_; }
            const TMatrixD& getN2() const { return N2_; }

            size_t numSolutions() const { return nunu_s.size(); }

            bool isValid(size_t idx = 0) const {
                return idx < nunu_s.size() && isFinitePair(nunu_s[idx]);
            }

            bool hasValidSolution() const { return isValid(); }

            double nu1_px(size_t idx = 0) const {
                return idx < nunu_s.size() ? nunu_s[idx].first[0]
                                           : std::numeric_limits<double>::quiet_NaN();
            }

            double nu1_py(size_t idx = 0) const {
                return idx < nunu_s.size() ? nunu_s[idx].first[1]
                                           : std::numeric_limits<double>::quiet_NaN();
            }

            double nu2_px(size_t idx = 0) const {
                return idx < nunu_s.size() ? nunu_s[idx].second[0]
                                           : std::numeric_limits<double>::quiet_NaN();
            }

            double nu2_py(size_t idx = 0) const {
                return idx < nunu_s.size() ? nunu_s[idx].second[1]
                                           : std::numeric_limits<double>::quiet_NaN();
            }

            bool usedMinimizerFallback() const { return usedMinimizerFallback_; }

        private:
            TMatrixD H1, H2; // store the ellipse matrices of the selected pairing
            TMatrixD N1_;
            TMatrixD N2_;
            bool usedMinimizerFallback_ = false;

            struct PairingResult {
                std::vector<NuPair> solutions;
                TMatrixD H1;
                TMatrixD H2;
                TMatrixD N1;
                TMatrixD N2;
                bool usedMinimizerFallback;

                PairingResult()
                    : solutions(), H1(3, 3), H2(3, 3), N1(3, 3), N2(3, 3), usedMinimizerFallback(false)
                {
                    H1.Zero();
                    H2.Zero();
                    N1.Zero();
                    N2.Zero();
                }
            };

            static bool isFinitePair(const NuPair& pair) {
                return std::isfinite(pair.first[0]) &&
                       std::isfinite(pair.first[1]) &&
                       std::isfinite(pair.second[0]) &&
                       std::isfinite(pair.second[1]);
            }

            static double metResidual(const PairingResult& res, double met_x, double met_y) {
                if (res.solutions.empty()) {
                    return std::numeric_limits<double>::infinity();
                }
                const auto& best = res.solutions.front();
                double sumx = best.first[0] + best.second[0];
                double sumy = best.first[1] + best.second[1];
                double residual = std::hypot(sumx - met_x, sumy - met_y);
                if (debug_enabled()) {
                    debug_log("doubleNeutrinoSolution::metResidual: sumx=" + std::to_string(sumx) +
                              ", sumy=" + std::to_string(sumy) +
                              ", met_x=" + std::to_string(met_x) +
                              ", met_y=" + std::to_string(met_y) +
                              ", residual=" + std::to_string(residual));
                }
                return residual;
            }

            std::vector<NuPair> nunu_s;
        };
        } // namespace nuana
       
        // ---------- Aliases ----------
        using nuana::nuSolutionSet;
        using nuana::singleNeutrinoSolution;
        using nuana::doubleNeutrinoSolution;
        using nuana::makeMetCov;
        
        """)

        # Define b-jet selection criteria
        ROOT.gInterpreter.Declare("""
std::vector<int> get_bjet_indices(const RVec<Float_t>& Jet_btagDeepFlavB,
                                  const RVec<Float_t>& CleanJet_eta,
                                  const RVec<Float_t>& CleanJet_pt,
                                  const RVec<int>& CleanJet_jetIdx) {
    std::vector<int> bjet_indices;
    for (size_t i = 0; i < CleanJet_pt.size(); ++i) {
        if (CleanJet_pt[i] > 30 && std::abs(CleanJet_eta[i]) < 2.5 && Jet_btagDeepFlavB[CleanJet_jetIdx[i]] > 0.0583)
            bjet_indices.push_back(i);
    }
    return bjet_indices;
    }
        """)



        df = df.Define(
            "bjet_indices",
            "get_bjet_indices(Jet_btagDeepFlavB, CleanJet_eta, CleanJet_pt, CleanJet_jetIdx)",
        )
        df = df.Define(
            "pass_bjets",
            "(bjet_indices.size() >= 2) && (Lepton_pt.size() >= 2) && ((nMuon + nElectron) >= 2)",
        )

        monitor_df = df.Filter("pass_bjets")
        passed_events = monitor_df.Count()
        total_events = df.Count()
        values.append([
            format_cutflow,
            "tt::twoBjets_twoLeptons",
            passed_events,
            total_events,
        ])

        df = df.Define("pass_bjets_float", "pass_bjets ? 1.0 : 0.0")
        df = df.Define(
            "b1",
            "TLorentzVector b1; if (!pass_bjets || bjet_indices.size() < 1) return b1; "
            "b1.SetPtEtaPhiM(CleanJet_pt[bjet_indices[0]], CleanJet_eta[bjet_indices[0]], "
            "CleanJet_phi[bjet_indices[0]], CleanJet_mass[bjet_indices[0]]); return b1;",
        )
        df = df.Define(
            "b2",
            "TLorentzVector b2; if (!pass_bjets || bjet_indices.size() < 2) return b2; "
            "b2.SetPtEtaPhiM(CleanJet_pt[bjet_indices[1]], CleanJet_eta[bjet_indices[1]], "
            "CleanJet_phi[bjet_indices[1]], CleanJet_mass[bjet_indices[1]]); return b2;",
        )
        df = df.Define(
            "l1",
            "TLorentzVector l1; if (!pass_bjets || Lepton_pt.size() < 1) return l1; "
            "l1.SetPtEtaPhiM(Lepton_pt[0], Lepton_eta[0], Lepton_phi[0], Lepton_mass[0]); return l1;",
        )
        df = df.Define(
            "l2",
            "TLorentzVector l2; if (!pass_bjets || Lepton_pt.size() < 2) return l2; "
            "l2.SetPtEtaPhiM(Lepton_pt[1], Lepton_eta[1], Lepton_phi[1], Lepton_mass[1]); return l2;",
        )
        df = df.Define("met_x", "PuppiMET_pt * TMath::Cos(PuppiMET_phi)")
        df = df.Define("met_y", "PuppiMET_pt * TMath::Sin(PuppiMET_phi)")

#        covariance_sources = [
#            ("PuppiMET_covXX", "PuppiMET_covXY", "PuppiMET_covYY"),
#            ("MET_covXX", "MET_covXY", "MET_covYY"),
#        ]
#        cov_expr = "makeMetCov(1.0, 0.0, 1.0)"
#        for cov_xx, cov_xy, cov_yy in covariance_sources:
#            if all(df.df.HasColumn(col) for col in (cov_xx, cov_xy, cov_yy)):
#                cov_expr = f"makeMetCov({cov_xx}, {cov_xy}, {cov_yy})"
#                break
       
        # Define leptons momenta in x and y
        df = df.Define("l1_pt_x", "pass_bjets ? l1.Px() : -999.0")
        df = df.Define("l1_pt_y", "pass_bjets ? l1.Py() : -999.0")
        df = df.Define("l1_phi", "pass_bjets ? l1.Phi() : -999.0")
        df = df.Define("l2_pt_x", "pass_bjets ? l2.Px() : -999.0")
        df = df.Define("l2_pt_y", "pass_bjets ? l2.Py() : -999.0")
        df = df.Define("l2_phi", "pass_bjets ? l2.Phi() : -999.0")
        # Define b-jet momenta in x and y
        df = df.Define("b1_pt_x",  "pass_bjets ? b1.Px() : -999.0")
        df = df.Define("b1_pt_y",  "pass_bjets ? b1.Py() : -999.0")
        df = df.Define("b1_phi",  "pass_bjets ? b1.Phi() : -999.0")
        df = df.Define("b2_pt_x",  "pass_bjets ? b2.Px() : -999.0")
        df = df.Define("b2_pt_y",  "pass_bjets ? b2.Py() : -999.0")
        df = df.Define("b2_phi",  "pass_bjets ? b2.Phi() : -999.0")


#        # Build single-neutrino solutions for all b/lepton pairings
#        single_pairs = [
#            ("b1l1", "b1", "l1"),
#            ("b1l2", "b1", "l2"),
#            ("b2l1", "b2", "l1"),
#            ("b2l2", "b2", "l2"),
#        ]
#
#        for suffix, b_name, l_name in single_pairs:
#            sn_name = f"snsol_{suffix}"
#            df = df.Define(
#                sn_name,
#                f"pass_bjets ? singleNeutrinoSolution({b_name}, {l_name}, met_x, met_y, {cov_expr}) : singleNeutrinoSolution()",
#            )
#            df = df.Define(
#                f"singleNu_{suffix}_valid",
#                f"pass_bjets ? {sn_name}.isValid() : false",
#            )
#            df = df.Define(
#                f"singleNu_{suffix}_chi2",
#                f"pass_bjets ? {sn_name}.chi2() : -999.0",
#            )
#            df = df.Define(
#                f"singleNu_{suffix}_px",
#                f"pass_bjets ? {sn_name}.nu_px() : -999.0",
#            )
#            df = df.Define(
#                f"singleNu_{suffix}_py",
#                f"pass_bjets ? {sn_name}.nu_py() : -999.0",
#            )
#            df = df.Define(
#                f"singleNu_{suffix}_pz",
#                f"pass_bjets ? {sn_name}.nu_pz() : -999.0",
#            )
#            df = df.Define(
#                f"singleNu_{suffix}_pt",
#                f"pass_bjets ? {sn_name}.nu_pt() : -999.0",
#            )
#            df = df.Define(
#                f"singleNu_{suffix}_phi",
#                f"pass_bjets ? {sn_name}.nu_phi() : -999.0",
#            )
#            df = df.Define(
#                f"singleNu_{suffix}_energy",
#                f"pass_bjets ? {sn_name}.nu_energy() : -999.0",
#            )
#            df = df.Define(
#                f"singleNu_{suffix}_usedMatrixFallback",
#                f"pass_bjets ? {sn_name}.usedMatrixFallback() : false",
#            )

        # Build double-neutrino solutions
        df = df.Define(
            "dnsol",
            "pass_bjets ? doubleNeutrinoSolution(b1, b2, l1, l2, met_x, met_y) : doubleNeutrinoSolution()",
        )

        df = df.Define(
            "H1_flat",
            "pass_bjets ? std::vector<double>(dnsol.getH1().GetMatrixArray(), dnsol.getH1().GetMatrixArray() + 9) : std::vector<double>(9, -999.0)",
        )
        df = df.Define(
            "H2_flat",
            "pass_bjets ? std::vector<double>(dnsol.getH2().GetMatrixArray(), dnsol.getH2().GetMatrixArray() + 9) : std::vector<double>(9, -999.0)",
        )

        df = df.Define(
            "N1_flat",
            "pass_bjets ? std::vector<double>(dnsol.getN1().GetMatrixArray(), dnsol.getN1().GetMatrixArray() + 9) : std::vector<double>(9, -999.0)",
        )

        df = df.Define(
            "N2_flat",
            "pass_bjets ? std::vector<double>(dnsol.getN2().GetMatrixArray(), dnsol.getN2().GetMatrixArray() + 9) : std::vector<double>(9, -999.0)",
        )

        df = df.Define("nu1_px", "pass_bjets ? dnsol.nu1_px() : -999.0")
        df = df.Define("nu1_py", "pass_bjets ? dnsol.nu1_py() : -999.0")
        df = df.Define("nu2_px", "pass_bjets ? dnsol.nu2_px() : -999.0")
        df = df.Define("nu2_py", "pass_bjets ? dnsol.nu2_py() : -999.0")

        df = df.Define(
            "dnsol_usedMinimizerFallback",
            "pass_bjets ? dnsol.usedMinimizerFallback() : false",
        )

        df = df.Define("ttbarReco_success", "pass_bjets && dnsol.hasValidSolution()")

        df = df.Define(
            "has_valid_nunu",
            "pass_bjets && dnsol.hasValidSolution()",
        )

        # Reconstruct tops
        df = df.Define(
            "top1",
            "TLorentzVector top1; if (!pass_bjets) return top1; TLorentzVector tmp(nu1_px, nu1_py, 0, sqrt(nu1_px*nu1_px + nu1_py*nu1_py));"
            " top1 = tmp; top1 += b1; top1 += l1; return top1;",
        )
        df = df.Define(
            "top2",
            "TLorentzVector top2; if (!pass_bjets) return top2; TLorentzVector tmp(nu2_px, nu2_py, 0, sqrt(nu2_px*nu2_px + nu2_py*nu2_py));"
            " top2 = tmp; top2 += b2; top2 += l2; return top2;",
        )

        # Leptons in top rest frames
        df = df.Define(
            "l1_top_rf",
            "if (!pass_bjets) return TVector3(); auto l1v = l1; l1v.Boost(-top1.BoostVector()); return l1v.Vect().Unit();",
        )
        df = df.Define(
            "l2_top_rf",
            "if (!pass_bjets) return TVector3(); auto l2v = l2; l2v.Boost(-top2.BoostVector()); return l2v.Vect().Unit();",
        )

        # Cosine between leptons in top rest frames
        df = df.Define("chel", "pass_bjets ? l1_top_rf.Dot(l2_top_rf) : -999.0")

        # Absolute Δφ between tops
        df = df.Define(
            "dphi_ttbar",
            "pass_bjets ? fabs(TVector2::Phi_mpi_pi(top1.Phi() - top2.Phi())) : -999.0",
        )

        # MET residual
        df = df.Define(
            "pdark",
            "pass_bjets ? ((met_x - nu1_px - nu2_px)*(met_x - nu1_px - nu2_px) + (met_y - nu1_py - nu2_py)*(met_y - nu1_py - nu2_py)) : -999.0",
        );

        # Drop intermediate helper columns to keep the dataframe clean
        columns_to_drop = [
            "bjet_indices",
            "l1", "l2", "b1", "b2",
#            "snsol_b1l1", "snsol_b1l2", "snsol_b2l1", "snsol_b2l2",
            "dnsol",
            "top1", "top2", "l1_top_rf", "l2_top_rf",
            "pass_bjets_float",
            ]
        
        for col in columns_to_drop:
            df = df.DropColumns(col)

        plot_columns_float = [ "nu1_px", "nu1_py", "nu2_px", "nu2_py", "met_x", "met_y", "l1_pt_x", "l1_pt_y", "l2_pt_x", "l2_pt_y", "b1_pt_x", "b1_pt_y", "b2_pt_x", "b2_pt_y" ]

        plot_columns_vector = ["H1_flat", "H2_flat", "N1_flat", "N2_flat"]

        data_cache = {}

        for col in plot_columns_float:
            vals = df.df.Take[ROOT.double](col)  # lowercase 'double'
            data_cache[col] = list(vals)

        vec_double_type = ROOT.std.vector('double')
        for col in plot_columns_vector:
            vals = df.df.Take[vec_double_type](col)  # each vals[i] is already vector<double>
            data_cache[col] = list(vals)

        pass_mask = list(df.df.Take[ROOT.double]("pass_bjets_float"))
        n_events = len(pass_mask)

        for i in range(n_events):
            if pass_mask[i] < 0.5:
                continue
            if not (
                math.isfinite(data_cache["nu1_px"][i])
                and math.isfinite(data_cache["nu1_py"][i])
                and math.isfinite(data_cache["nu2_px"][i])
                and math.isfinite(data_cache["nu2_py"][i])
            ):
                continue

            plot_args = [data_cache[col][i] for col in plot_columns_float]
            plot_args += [data_cache[col][i] for col in plot_columns_vector]  # each is already vector<double>
            plot_args.append(i)
            plot_event(*plot_args)

        for col in ("H1_flat", "H2_flat", "N1_flat", "N2_flat"):
            df = df.DropColumns(col)

        # Return the final dataframe or continue processing
        return df
        


