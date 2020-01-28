// Linear algebra example. We start with a matrix.
// two by two
mAsstCor22  = [[1.0, 0.5],
               [0.5, 1.0]];
vAsstVol2   = [0.20, 0.30];
                                                                        // calc covriance from correlations and sigmas  
volxXvoly22 = numeric.dot(numeric.transpose([vAsstVol2]),[vAsstVol2]);  // calculate 2x2 matrix of sigma products  sigma_x X sigma_Y
mAsstCov22  = numeric.mul(volxXvoly22,mAsstCor22);                      // point-wise mutplications for // cov(x,y) = corr(x,y)*vol(x)*vol(y)
                                                                        //= [[0.04 0.03]
                                                                        //   [0.03 0.09]]
                                                                 
vOnes2        = numeric.rep([2],1);
mAsstCov22Inv = numeric.inv(mAsstCov22);

                        // minimum variance portfolio = (V^-1 x ones) / (ones' x V^-1 x ones)
vAsstWghts2minvar     = numeric.div(numeric.dot(mAsstCov22Inv, vOnes2) , numeric.rep([2],numeric.dot(vOnes2,numeric.dot(mAsstCov22Inv, vOnes2))))
vAsstWghts2minvar     = numeric.dot(mAsstCov22Inv, vOnes2) / numeric.dot(vOnes2,numeric.dot(mAsstCov22Inv, vOnes2))
vAsstWghts2bookanswer = [6/7, 1/7];
vAsstWghts2onlyA      = [1, 0];
vAsstWghts2onlyB      = [0, 1];
vAsstWghts2halfAhalfB = [1/2, 1/2];

sTotRisk = Math.sqrt(numeric.dot(vAsstWghts2onlyA,numeric.dot(mAsstCov22,vAsstWghts2onlyA)));

// three by three 
mAsstCor33   = [[1.0, 0.5, 0.5],
                [0.5, 1.0, 0.5],
                [0.5, 0.5, 1.0]];
vAsstVol3    = [0.20, 0.30, 0.1];
vAsstWghts3 = [0.5, 0.5, 0.0];

  
mAsstCov33 = numeric.dot(vAsstVol2,numeric.dot(mAsstCor22,vAsstVol2));  // cov(x,y) = corr(x,y)*vol(x)*vol(y)


// Let's also make a vector


// Matrix-vector product
sTotRisk = numeric.dot(vAsstWghts,numeric.dot(mAsstCov,vAsstWghts));
// b = numeric.dot(mAsstCov, vAsstWghts);
// Matrix inverse.
// Ainv = numeric.inv(A);
// Determinant
// numeric.det(A);
