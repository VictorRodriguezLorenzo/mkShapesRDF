import ROOT
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def flat_to_matrix_3x3(flat):
    """Convert flattened list/array of length 9 into a 3×3 matrix."""
    return [[flat[i*3 + j] for j in range(3)] for i in range(3)]

def parametrize_ellipse(H):
    """Eigen-decomposition of 2×2 submatrix [[H00, H01],[H01,H11]]."""
    a11, a12 = H[0][0], H[0][1]
    a22 = H[1][1]

    trace = a11 + a22
    det = a11*a22 - a12*a12
    temp = math.sqrt(max(0.0, trace*trace/4 - det))

    eval1 = trace/2 + temp
    eval2 = trace/2 - temp

    # Handle degeneracies
    if eval1 <= 0 or eval2 <= 0:
        return None

    if abs(a12) > 1e-12:
        vec1 = [eval1 - a22, a12]
    elif abs(a11 - eval1) > 1e-12:
        vec1 = [a12, eval1 - a11]
    else:
        vec1 = [1.0, 0.0]

    norm = math.sqrt(vec1[0]**2 + vec1[1]**2)
    vec1 = [v / norm for v in vec1]

    a = 1.0 / math.sqrt(eval1)
    b = 1.0 / math.sqrt(eval2)
    angle = math.degrees(math.atan2(vec1[1], vec1[0]))
    return a, b, angle

def plot_event(nu1_px, nu1_py,
               nu2_px, nu2_py,
               met_x, met_y,
               l1_pt_x, l1_pt_y,
               l2_pt_x, l2_pt_y,
               b1_pt_x, b1_pt_y,
               b2_pt_x, b2_pt_y,
               H1_flat, H2_flat,
               event_idx):

    fig, ax = plt.subplots(figsize=(8, 8))

    def arrow(x, y, label, color, lw=1.5):
        ax.arrow(0, 0, x, y, head_width=2, length_includes_head=True,
                 color=color, alpha=0.8, linewidth=lw)
        ax.plot([], [], color=color, label=label)

    # Momentum vectors
    arrow(nu1_px, nu1_py, "Neutrino 1", 'blue')
    arrow(nu2_px, nu2_py, "Neutrino 2", 'pink')
    arrow(met_x, met_y, "MET", 'red', lw=2)
    arrow(l1_pt_x, l1_pt_y, "Lepton 1", 'lightblue')
    arrow(l2_pt_x, l2_pt_y, "Lepton 2", 'purple')
    arrow(b1_pt_x, b1_pt_y, "B-jet 1", 'cyan')
    arrow(b2_pt_x, b2_pt_y, "B-jet 2", 'magenta')

    # Ellipses
    H1 = flat_to_matrix_3x3(H1_flat)
    H2 = flat_to_matrix_3x3(H2_flat)
    e1 = parametrize_ellipse(H1)
    e2 = parametrize_ellipse(H2)

    if e1 is not None:
        a1, b1_, angle1 = e1
        ax.add_patch(patches.Ellipse((0, 0), 2*a1, 2*b1_, angle=angle1,
                                     color='blue', alpha=0.3))
    if e2 is not None:
        a2, b2_, angle2 = e2
        ax.add_patch(patches.Ellipse((0, 0), 2*a2, 2*b2_, angle=angle2,
                                     color='red', alpha=0.3))

    # Formatting
    ax.set_xlabel('pT_x (GeV)')
    ax.set_ylabel('pT_y (GeV)')
    ax.set_title(f'Event {event_idx}: Neutrino Solutions and Conic Ellipses')
    ax.axhline(0, color='black', lw=0.5, ls='--')
    ax.axvline(0, color='black', lw=0.5, ls='--')
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal')

    outdir = "/afs/cern.ch/user/v/victorr/private/mkShapesRDF/mkShapesRDF/processor/condor/Summer22_130x_nAODv12_Full2022v12/MCl1loose2022v12__fakeSel/TTTo2L2Nu_test/plots"
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(f"{outdir}/event_{event_idx}_neutrino_solutions.png")
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
        #include <algorithm>
        #include <limits>
        #include <iostream>
        #include <TMatrixDSymEigen.h>
                                  

        namespace nuana {

        // ---------- Utilities ----------
        // UnitCircle: returns a 3x3 matrix representing the unit circle in the F' coordinate system
        TMatrixD UnitCircle() {
            TMatrixD U(3,3);
            U.Zero();
            U(0,0) = 1.0;
            U(1,1) = 1.0;
            U(2,2) = -1.0;
            return U;
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
            double c = std::cos(angle);
            double s = std::sin(angle);

            TMatrixD I(3, 3);
            I.UnitMatrix();

            TMatrixD R = c * I;
            for (int i = -1; i <= 1; ++i) {
                int row = (axis - i + 3) % 3;
                int col = (axis + i + 3) % 3;
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

            // Direct check: G[0,0] == 0 and G[1,1] == 0
            if (G(0,0) == 0.0 && G(1,1) == 0.0) {
                lines.push_back({G(0,1), 0.0, G(1,2)});
                lines.push_back({0.0, G(0,1), G(0,2) - G(1,2)});
                return lines;
            }

            bool swapXY = std::abs(G(0,0)) > std::abs(G(1,1));

            // Build Q with possible index swap
            TMatrixD Q(3,3);
            if (swapXY) {
                int idx[3] = {1,0,2};
                for (int r=0; r<3; ++r)
                    for (int c=0; c<3; ++c)
                        Q(r,c) = G(idx[r], idx[c]);
            } else {
                Q = G;
            }

            // Always divide Q by Q(1,1)
            double denom = Q(1,1);
            for (int r=0; r<3; ++r)
                for (int c=0; c<3; ++c)
                    Q(r,c) /= denom;

            double q22 = cofactor(Q, 2, 2);

            if (-q22 <= zero) {
                double cof00 = cofactor(Q, 0, 0);
                auto sq = multisqrt(-cof00);
                for (double s : sq) {
                    lines.push_back({Q(0,1), Q(1,1), Q(1,2) + s});
                }
            } else {
                double x0 = cofactor(Q, 0, 2) / q22;
                double y0 = cofactor(Q, 1, 2) / q22;
                auto ms = multisqrt(-q22);
                for (double s : ms) {
                    double m = Q(0,1) + s;
                    lines.push_back({m, Q(1,1), -Q(1,1)*y0 - m*x0});
                }
            }

            // Final swap if swapXY was true
            if (swapXY) {
                for (auto &L : lines) {
                    std::swap(L[0], L[1]);
                }
            }

            return lines;
        }


        std::vector<double> intersections_ellipse_line(
            const TMatrixD &ellipse,
            const TVectorD &line,
            double zero = 1e-12
        ) {
            // Cross product matrix: np.cross(line, ellipse) in matrix form
            TMatrixD C(3,3);
            C.Zero();
            C(0,1) = -line[2];
            C(0,2) =  line[1];
            C(1,0) =  line[2];
            C(1,2) = -line[0];
            C(2,0) = -line[1];
            C(2,1) =  line[0];

            // CE = cross(line, ellipse)
            TMatrixD CE = C * ellipse;

            // M = (CE)^T
            TMatrixD M(TMatrixD::kTransposed, CE);

            // Eigen decomposition (possibly complex)
            TMatrixDEigen eig(M);
            TVectorD evalsRe = eig.GetEigenValuesRe(); // not actually needed here
            TVectorD evalsIm = eig.GetEigenValuesIm();
            const TMatrixD &eigVecsRe = eig.GetEigenVectors(); // only real parts available

            // Store (s, k) pairs
            std::vector<std::pair<double,double>> sols;

            for (int i = 0; i < 3; ++i) {
                TVectorD v(3);
                for (int j = 0; j < 3; ++j) {
                    v[j] = eigVecsRe(j, i); // Python's v.real
                }

                // In Python: s = v.real / v[2].real
                if (std::abs(v[2]) < 1e-15) continue;
                double s = v[0] / v[2];

                // lv = np.dot(line, v.real)
                double lv = line[0]*v[0] + line[1]*v[1] + line[2]*v[2];

                // vev = np.dot(v.real, ellipse.dot(v.real))
                TVectorD Ev = ellipse * v;
                double vev = v[0]*Ev[0] + v[1]*Ev[1] + v[2]*Ev[2];

                // k = lv**2 + vev**2
                double k = lv*lv + vev*vev;

                sols.push_back({s, k});
            }

            // Sort by k, take first 2
            std::sort(sols.begin(), sols.end(),
                    [](const auto &a, const auto &b){ return a.second < b.second; });

            if (sols.size() > 2) sols.resize(2);

            // Keep only those with k < zero
            std::vector<double> result;
            for (auto &p : sols) {
                if (p.second < zero) {
                    result.push_back(p.first);
                }
            }

            return result;
        }

        
                                  
        std::pair<std::vector<TVectorD>, std::vector<std::array<double,3>>>
        intersections_ellipses(
            const TMatrixD &A, const TMatrixD &B, 
            bool returnLines = false, double zero = 1e-10
        ) {
            // Swap if needed to make |det(A)| >= |det(B)|
            double detA = A.Determinant();
            double detB = B.Determinant();
            const TMatrixD *AA = &A;
            const TMatrixD *BB = &B;
            if (std::abs(detB) > std::abs(detA)) {
                AA = &B;
                BB = &A;
            }

            // Compute inv(A) * B
            TMatrixD invA(*AA);
            invA.Invert();
            TMatrixD M = invA * (*BB);

            // Eigenvalues (possibly complex)
            TMatrixDEigen eig(M);
            TVectorD evalsRe = eig.GetEigenValuesRe();
            TVectorD evalsIm = eig.GetEigenValuesIm();
                                  

            // Pick the first purely real eigenvalue
            double e_found = 0.0;
            bool found = false;
            for (int i = 0; i < evalsRe.GetNrows(); ++i) {
                if (std::abs(evalsIm(i)) < zero) {
                    e_found = evalsRe(i);
                    found = true;
                    break;
                }
            }
            if (!found) {
                throw std::runtime_error("No purely real eigenvalue found.");
            }

            // Degenerate conic
            TMatrixD G = (*BB) - e_found * (*AA);

            // Factor degenerate conic into lines
            auto lines = factor_degenerate(G); // should return vector<array<double,3>>

            // Collect intersection points
            std::vector<TVectorD> points;
            for (const auto &L : lines) {
                // intersections_ellipse_line should return vector<TVectorD> directly
                TVectorD line(3);
                line(0) = L[0];
                line(1) = L[1];
                line(2) = L[2];
                auto pts = intersections_ellipse_line(*AA, line, zero);
                points.insert(points.end(), pts.begin(), pts.end());
            }
            cout << "Found " << points.size() << " intersection points." << std::endl;

            if (returnLines) {
                return {points, lines};
            } else {
                return {points, {}}; // empty line list
            }
        }       

        struct nuSolutionSet {
            TLorentzVector b, mu;
            double c, s, x0, x0p, Sx, Sy, w, w_, x1, y1, Z, Om2, eps2, mW2;

            nuSolutionSet(const TLorentzVector& b_, const TLorentzVector& mu_,
                  double mW = 80.385, double mT = 172.5, double mN = 0.0)
            : b(b_), mu(mu_) {
            mW2 = mW * mW;
            double mT2 = mT * mT;
            double mN2 = mN * mN;

            c = ROOT::Math::VectorUtil::CosTheta(b, mu);
            s = std::sqrt(std::max(0.0, 1.0 - c * c));

            x0p = - (mT2 - mW2 - b.M2()) / (2.0 * b.E());
            x0  = - (mW2 - mu.M2() - mN2) / (2.0 * mu.E());

            double Bb = b.Beta();
            double Bm = mu.Beta();

            Sx = (x0 * Bm - mu.P() * (1.0 - Bm * Bm)) / (Bm * Bm);
            Sy = (x0p / Bb - c * Sx) / s;

            w  = (Bm / Bb - c) / s;
            w_ = (-Bm / Bb - c) / s;

            Om2 = w * w + 1.0 - Bm * Bm;
            eps2 = (mW2 - mN2) * (1.0 - Bm * Bm);

            x1 = Sx - (Sx + w * Sy) / Om2;
            y1 = Sy - (Sx + w * Sy) * w / Om2;

            double Z2 = x1 * x1 * Om2 - (Sy - w * Sx) * (Sy - w * Sx) - (mW2 - x0 * x0 - eps2);
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
                double B2 = b.Beta() * b.Beta();
                double SxB2 = Sx * B2;
                double F = mW2 - x0 * x0 - eps2;

                A(0,0) = 1 - B2; A(0,1) = 0;    A(0,2) = 0; A(0,3) = SxB2;
                A(1,0) = 0;      A(1,1) = 1;    A(1,2) = 0; A(1,3) = 0;
                A(2,0) = 0;      A(2,1) = 0;    A(2,2) = 1; A(2,3) = 0;
                A(3,0) = SxB2;   A(3,1) = 0;    A(3,2) = 0; A(3,3) = F;

                return A;
            }

                                  
            // F coord. constraint on W momentum: ellipsoid
            TMatrixD A_b() const {
                TMatrixD K = getK();
                double B = b.Beta();
                double mw2 = mW2;
                double x0p = x0p;
           
                TMatrixD A_b_(4, 4);
                A_b_(0,0) = 1 - B*B;  A_b_(0,1) = 0;      A_b_(0,2) = 0;      A_b_(0,3) = B*x0p;
                A_b_(1,0) = 0;         A_b_(1,1) = 1;      A_b_(1,2) = 0;      A_b_(1,3) = 0;
                A_b_(2,0) = 0;         A_b_(2,1) = 0;      A_b_(2,2) = 1;      A_b_(2,3) = 0;
                A_b_(3,0) = B*x0p;     A_b_(3,1) = 0;      A_b_(3,2) = 0;      A_b_(3,3) = mW2 - x0p*x0p;
                return K*(A_b_)*K.T();
            }
            
                                  
            // Rotation from F coord. to laboratory coord.
            TMatrixD getR_T() const {
                TVector3 b_xyz = b.Vect();
                TMatrixD Rz = Rotation(2, -mu.Phi());
                TMatrixD Ry = Rotation(1, 0.5*M_PI - mu.Theta());
                TVector3 v = Ry * (Rz * b_xyz);
                TMatrixD Rx = Rotation(0, -std::atan2(v(2), v(1)));
                //cout << "Rz:" << endl; Rz.Print();
                //cout << "Ry:" << endl; Ry.Print();
                //cout << "Rx:" << endl; Rx.Print();
                // Return the combined rotation matrix
                return Rz * (Ry.T() * Rx.T());
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
            // Equivalent to: return self.R_T.dot(self.H_tilde)
            TMatrixD rt = getR_T();
            TMatrixD ht = getH_tilde();
            //cout << "R_T: " << std::endl;
            //rt.Print();
            //cout << "H_tilde: " << std::endl;
            //ht.Print();
            // Perform matrix multiplication
            return rt * ht;
            }

            // Transformation of t=[c,s,1] to pT_nu: lab coord.
            TMatrixD getH_perp() const {
            TMatrixD h = getH();
            TMatrixD h_perp(3,3);
            for (int i=0; i<2; ++i)
            for (int j=0; j<3; ++j)
                h_perp(i,j) = h(i,j);
            h_perp(2,0) = 0.0;
            h_perp(2,1) = 0.0;
            h_perp(2,2) = 1.0;
            //cout << "H_perp: " << std::endl;
            //h_perp.Print();
            return h_perp;
            }
                                  

            // Solution ellipse of pT_nu: lab coord.
            TMatrixD getN() const {
                // Get the inverse of the H_perp matrix
                TMatrixD HpInv = getH_perp();
                //cout << "H_perp inverse: " << std::endl;
                //HpInv.Print();
                HpInv.Invert();
                //cout << "H_perp inverse after Invert: " << std::endl;
                //HpInv.Print();
                // Return the transformed ellipse
                return HpInv.T() * UnitCircle() * HpInv;
            }
        };

        // ---------- Classes ----------
        // singleNeutrinoSolution: finds the best single-neutrino solution for given b-jet and lepton momenta.
        class singleNeutrinoSolution {
        public:
            nuana::nuSolutionSet solutionSet; 
            TMatrixD X;
            std::vector<TVectorD> solutions;

            singleNeutrinoSolution(const TLorentzVector& b, const TLorentzVector& mu,
                                  double metX, double metY,
                                  const TMatrixD& sigma2,
                                  double mW2 = 80.385*80.385, double mT2 = 172.5*172.5)
                : solutionSet(b, mu, std::sqrt(mW2), std::sqrt(mT2))
            {
                // Build S2: inverse of sigma2, padded to 3x3
                TMatrixD S2(3,3); S2.Zero();
                TMatrixD sigma2_inv(sigma2); sigma2_inv.Invert();
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

                // M = X * Derivative() + (X * Derivative()).T
                TMatrixD XD = X * Derivative();
                TMatrixD M = XD;
                TMatrixD XD_T(TMatrixD::kTransposed, XD);
                for (int i=0;i<3;++i) for (int j=0;j<3;++j)
                    M(i,j) += XD_T(i,j);

                // Find intersections
                solutions = intersections_ellipses(M, UnitCircle()).first;


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
                if (solutions.empty()) return -1.0;
                return calcX2(solutions[0]);
            }

            TVectorD nu() const {
                if (solutions.empty()) return TVectorD(3);
                return solutionSet.getH() * solutions[0];
            }
        };
                                  

        // doubleNeutrinoSolution: finds the best double-neutrino solution for given b-jet and lepton momenta.
        class doubleNeutrinoSolution {
        private:
            TMatrixD H1, H2; // store the ellipse matrices
        public:
            struct NuPair {
                std::array<double, 2> first;
                std::array<double, 2> second;
            };

            std::vector<NuPair> nunu_s;

            // default constructor
            doubleNeutrinoSolution() = default;

            doubleNeutrinoSolution(
                const TLorentzVector& b1,
                const TLorentzVector& b2,
                const TLorentzVector& l1,
                const TLorentzVector& l2,
                double met_x,
                double met_y)
            {
                double mW = 80.385; // W mass in GeV
                double mT = 172.5;  // Top mass in GeV

                // Try both pairings: (b1,l1)+(b2,l2) and (b1,l2)+(b2,l1)
                auto try_pairing = [&](const TLorentzVector& B1, const TLorentzVector& B2,
                                    const TLorentzVector& L1, const TLorentzVector& L2) {
                    std::vector<NuPair> solutions;

                    // Build solution sets
                    nuana::nuSolutionSet ss1(B1, L1, mW, mT);
                    nuana::nuSolutionSet ss2(B2, L2, mW, mT);

                    // Build the S matrix
                    TMatrixD V0(3, 3);
                    V0.Zero();
                    V0(0, 2) = met_x;
                    V0(1, 2) = met_y;
                    V0(2, 2) = 0.0;

                    // S = V0 - UnitCircle();
                    TMatrixD S = V0 - UnitCircle();
                    
                    // Save ellipses for plotting
                    H1 = ss1.getH();
                    H2 = ss2.getH();

                    // Get solution ellipses
                    TMatrixD N1 = ss1.getN();
                    TMatrixD N2 = ss2.getN();

                    // Rotate N2 to frame of N1
                    TMatrixD n2 = S.T() * N2 * S;


                    // Find intersection points
                    std::vector<TVectorD> v = nuana::intersections_ellipses(N1, n2).first;

                    for (const auto& sol : v) {
                        // Neutrino 1
                        TVectorD nu1 = ss1.getH() * sol;
                        // Neutrino 2
                        TVectorD nu2 = S * sol;

                        //cout << "Neutrino 1: " << nu1(0) << ", " << nu1(1) << std::endl;
                        //cout << "Neutrino 2: " << nu2(0) << ", " << nu2(1) << std::endl;

                        NuPair pair;
                        pair.first = {nu1(0), nu1(1)};
                        pair.second = {nu2(0), nu2(1)};
                        // Check if both neutrinos are valid (not NaN)
                        if (std::isfinite(pair.first[0]) && std::isfinite(pair.first[1]) &&
                            std::isfinite(pair.second[0]) && std::isfinite(pair.second[1])) {
                            //cout << "Valid solution: " << pair.first[0] << ", " << pair.first[1] << " | "
                            //     << pair.second[0] << ", " << pair.second[1] << std::endl;
                            //    solutions.push_back(pair);
                        } else {
                            //cout << "Invalid solution: " << pair.first[0] << ", " << pair.first[1] << " | "
                            //     << pair.second[0] << ", " << pair.second[1] << std::endl;
                        }
                    }

                    // If no solutions or only one, use fallback
                    if (solutions.empty() || solutions.size() < 2) {
                        solutions.clear();
                        cout << "No valid solutions found, trying fallback method." << std::endl;

                        TMatrixD es1 = ss1.getH_perp();
                        TMatrixD es2 = ss2.getH_perp();
                        TVectorD met(3);
                        met(0) = met_x;
                        met(1) = met_y;
                        met(2) = 1.0;

                        // Function to compute neutrino momenta (3D vector each)
                        auto nus = [&](const TVectorD& ts) {
                            std::vector<TVectorD> result;
                            for (int i = 0; i < 2; ++i) {
                                double t = ts(i);
                                TVectorD vec(3);
                                vec(0) = std::cos(t);
                                vec(1) = std::sin(t);
                                vec(2) = 1.0;

                                TVectorD nu(3);
                                if (i == 0) nu = es1 * vec;
                                else        nu = es2 * vec;

                                result.push_back(nu);
                            }
                            return result;
                        };

                        // Residuals: px, py only
                        auto residuals = [&](const TVectorD& params) {
                            auto nu_vecs = nus(params);
                            TVectorD total = nu_vecs[0] + nu_vecs[1] - met;

                            TVectorD res(2);
                            res(0) = total(0);
                            res(1) = total(1);
                            return res;
                        };

                        // Minimizer
                        ROOT::Math::Minimizer* min =
                            ROOT::Math::Factory::CreateMinimizer("Minuit2", "Migrad");
                        min->SetTolerance(1e-10);
                        min->SetPrecision(1e-12); 
                        min->SetVariableStepSize(0, 0.01);
                        min->SetVariableStepSize(1, 0.01);

                        // Wrapper class
                        class ResidualsFunction : public ROOT::Math::IMultiGenFunction {
                        public:
                            ResidualsFunction(const std::function<TVectorD(const TVectorD&)>& f) : func(f) {}
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

                        ResidualsFunction residualsFunc(residuals);
                        min->SetFunction(residualsFunc);

                        min->SetVariable(0, "t1", 0.0, 0.1);
                        min->SetVariable(1, "t2", 0.0, 0.1);

                        min->Minimize();

                        TVectorD ts(2);
                        ts(0) = min->X()[0];
                        ts(1) = min->X()[1];

                        auto v = nus(ts);

                        NuPair pair;
                        pair.first = {v[0](0), v[0](1)};
                        pair.second = {v[1](0), v[1](1)};
                        solutions.push_back(pair);
                    }

                    return solutions;
                };

                // Try both pairings and pick the one with smaller MET residual
                auto sol1 = try_pairing(b1, b2, l1, l2);
                auto sol2 = try_pairing(b1, b2, l2, l1);

                auto met_residual = [&](const std::vector<NuPair>& sols) {
                    if (sols.empty()) return 1e9;
                    double sumx = sols[0].first[0] + sols[0].second[0];
                    double sumy = sols[0].first[1] + sols[0].second[1];
                    return std::hypot(sumx - met_x, sumy - met_y);
                };

                //cout << "MET residuals: " << met_residual(sol1) << ", " << met_residual(sol2) << std::endl;
                if (met_residual(sol1) <= met_residual(sol2))
                    nunu_s = sol1;
                else
                    nunu_s = sol2;
            }

            std::vector<NuPair> get_nunu_s() const {
                cout << "Best solution: " << (nunu_s.empty() ? "None" : std::to_string(nunu_s[0].first[0]) + ", " + std::to_string(nunu_s[0].first[1]) + " | " + std::to_string(nunu_s[0].second[0]) + ", " + std::to_string(nunu_s[0].second[1])) << std::endl;
                return nunu_s;
            }
            
            // --- NEW accessors for ellipse matrices ---
            const TMatrixD& getH1() const { return H1; }
            const TMatrixD& getH2() const { return H2; }
        };
        } // namespace nuana
       
        // ---------- Aliases ----------
        using nuana::nuSolutionSet;
        using nuana::singleNeutrinoSolution;
        using nuana::doubleNeutrinoSolution;
        
        """)

        # Define b-jet selection criteria
        ROOT.gInterpreter.Declare("""
std::vector<int> get_bjet_indices(const RVec<Float_t>& Jet_btagDeepFlavB,
                                  const RVec<Float_t>& CleanJet_eta,
                                  const RVec<Float_t>& CleanJet_pt,
                                  const RVec<int>& CleanJet_jetIdx) {
    std::vector<int> bjet_indices;
    for (size_t i = 0; i < CleanJet_pt.size(); ++i) {
        if (CleanJet_pt[i] > 30 && std::abs(CleanJet_eta[i]) < 2.5 && Jet_btagDeepFlavB[CleanJet_jetIdx[i]] > 0.1208)
            bjet_indices.push_back(i);
    }
    return bjet_indices;
    }
        """)



        df = df.Define("bjet_indices", "get_bjet_indices(Jet_btagDeepFlavB, CleanJet_eta, CleanJet_pt, CleanJet_jetIdx)")
        df = df.Filter("bjet_indices.size() >= 2 & (nMuon + nElectron) >= 2")
        df = df.Define("b1", "TLorentzVector b1; b1.SetPtEtaPhiM(CleanJet_pt[bjet_indices[0]], CleanJet_eta[bjet_indices[0]], CleanJet_phi[bjet_indices[0]], CleanJet_mass[bjet_indices[0]]); return b1;")
        df = df.Define("b2", "TLorentzVector b2; b2.SetPtEtaPhiM(CleanJet_pt[bjet_indices[1]], CleanJet_eta[bjet_indices[1]], CleanJet_phi[bjet_indices[1]], CleanJet_mass[bjet_indices[1]]); return b2;")
        df = df.Define("l1", "TLorentzVector l1; l1.SetPtEtaPhiM(Lepton_pt[0], Lepton_eta[0], Lepton_phi[0], 0.0); return l1;")
        df = df.Define("l2", "TLorentzVector l2; l2.SetPtEtaPhiM(Lepton_pt[1], Lepton_eta[1], Lepton_phi[1], 0.0); return l2;")
        df = df.Define("met_x", "PuppiMET_pt * TMath::Cos(PuppiMET_phi)")
        df = df.Define("met_y", "PuppiMET_pt * TMath::Sin(PuppiMET_phi)")
       
        # Define leptons momenta in x and y
        df = df.Define("l1_pt_x", "l1.Px()")
        df = df.Define("l1_pt_y", "l1.Py()")
        df = df.Define("l1_phi", "l1.Phi()")
        df = df.Define("l2_pt_x", "l2.Px()")
        df = df.Define("l2_pt_y", "l2.Py()")
        df = df.Define("l2_phi", "l2.Phi()")
        # Define b-jet momenta in x and y
        df = df.Define("b1_pt_x",  "b1.Px()")
        df = df.Define("b1_pt_y",  "b1.Py()")
        df = df.Define("b1_phi",  "b1.Phi()")
        df = df.Define("b2_pt_x",  "b2.Px()")
        df = df.Define("b2_pt_y",  "b2.Py()")
        df = df.Define("b2_phi",  "b2.Phi()")


        # Build double-neutrino solutions
        df = df.Define("dnsol", "doubleNeutrinoSolution(b1, b2, l1, l2, met_x, met_y)")
 
        df = df.Define("H1_flat", "std::vector<double>(dnsol.getH1().GetMatrixArray(), dnsol.getH1().GetMatrixArray() + 9)")
        df = df.Define("H2_flat", "std::vector<double>(dnsol.getH2().GetMatrixArray(), dnsol.getH2().GetMatrixArray() + 9)")

        # Extract the first MET-conserving neutrino pair
        df = df.Define("nu1_px", "dnsol.get_nunu_s().at(0).first[0]")
        df = df.Define("nu1_py", "dnsol.get_nunu_s().at(0).first[1]")
        df = df.Define("nu2_px", "dnsol.get_nunu_s().at(0).second[0]")
        df = df.Define("nu2_py", "dnsol.get_nunu_s().at(0).second[1]")

        # Reconstruct tops
        df = df.Define("top1", "TLorentzVector top1(nu1_px, nu1_py, 0, sqrt(nu1_px*nu1_px + nu1_py*nu1_py)); top1 += b1; top1 += l1; return top1;")
        df = df.Define("top2", "TLorentzVector top2(nu2_px, nu2_py, 0, sqrt(nu2_px*nu2_px + nu2_py*nu2_py)); top2 += b2; top2 += l2; return top2;")
        
        # Leptons in top rest frames
        df = df.Define("l1_top_rf", "auto l1v = l1; l1v.Boost(-top1.BoostVector()); return l1v.Vect().Unit();")
        df = df.Define("l2_top_rf", "auto l2v = l2; l2v.Boost(-top2.BoostVector()); return l2v.Vect().Unit();")
        
        # Cosine between leptons in top rest frames
        df = df.Define("chel", "l1_top_rf.Dot(l2_top_rf)")
        
        # Absolute Δφ between tops
        df = df.Define("dphi_ttbar", "fabs(TVector2::Phi_mpi_pi(top1.Phi() - top2.Phi()))")
        
        # MET residual
        df = df.Define("pdark", "(met_x - nu1_px - nu2_px)*(met_x - nu1_px - nu2_px) + (met_y - nu1_py - nu2_py)*(met_y - nu1_py - nu2_py)");

        # Drop intermediate helper columns to keep the dataframe clean
        columns_to_drop = [
            "bjet_indices", 
            "l1", "l2", "b1", "b2",
            "dnsol",
            "top1", "top2", "l1_top_rf", "l2_top_rf",
            ]
        
        for col in columns_to_drop:
            df = df.DropColumns(col)

        plot_columns_float = [ "nu1_px", "nu1_py", "nu2_px", "nu2_py", "met_x", "met_y", "l1_pt_x", "l1_pt_y", "l2_pt_x", "l2_pt_y", "b1_pt_x", "b1_pt_y", "b2_pt_x", "b2_pt_y" ]

        plot_columns_vector = ["H1_flat", "H2_flat"]

        data_cache = {}

        for col in plot_columns_float:
            vals = df.df.Take[ROOT.double](col)  # lowercase 'double'
            data_cache[col] = list(vals)

        vec_double_type = ROOT.std.vector('double')
        for col in plot_columns_vector:
            vals = df.df.Take[vec_double_type](col)  # each vals[i] is already vector<double>
            data_cache[col] = list(vals)

        n_events = len(data_cache[plot_columns_float[0]])

        for i in range(n_events):
           plot_args = [data_cache[col][i] for col in plot_columns_float]
           plot_args += [data_cache[col][i] for col in plot_columns_vector]  # each is already vector<double>
           plot_args.append(i)
           plot_event(*plot_args) 
        
        # Return the final dataframe or continue processing
        return df
        


