#include <math.h>

class Vector {
    private:
        double px;
        double py;
        double pz;
    public:
        /*
        double Px() {return px;};
        double Py() {return py;};
        double Pz() {return pz;};
        double Pt() {return sqrt(px*px + py*py);};
        double P()  {return sqrt(px*px + py*py + pz*pz);};
        double P2() {return px*px + py*py + pz*pz;};
        */
        void SetPtEtaPhi(double pt, double phi, double eta) {
            pt = fabs(pt);
            px = pt * cos(phi);
            py = pt * sin(phi);
            pz = pt * sinh(eta);
        };
        void SetXYZ(double Px, double Py, double Pz) {
            px = Px;
            py = Py;
            pz = Pz;
        };
};
