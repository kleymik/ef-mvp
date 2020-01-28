// Portfolio, risk, return and minimum variance

// two by two
vOnes2        = numeric.rep([2],1);
mAsstCor22    = [[1.0, 0.5],
                 [0.5, 1.0]];
vAsstVol2     = [0.30, 0.20];
                                                                           // calc covariance from correlations and sigmas  
mVolxXvoly22  = numeric.dot(numeric.transpose([vAsstVol2]), [vAsstVol2]);  // calculate 2x2 matrix of sigma products  sigma_x X sigma_Y
mAsstCov22    = numeric.mul(mVolxXvoly22, mAsstCor22);                     // point-wise mutplications for // cov(x,y) = corr(x,y)*vol(x)*vol(y)
mAsstCov22Inv = numeric.inv(mAsstCov22);                                   //= [[0.04 0.03]    inv=  1/27 * [[ 0.09 - 0.03]
                                                                           //   [0.03 0.09]]                 [-0.03   0.04]]

vAsstWghts2minvar     = numeric.div(numeric.dot(mAsstCov22Inv, vOnes2),    // minimum variance portfolio = (V^-1 x ones) / (ones' x V^-1 x ones)
                                    numeric.rep([2],numeric.dot(vOnes2, numeric.dot(mAsstCov22Inv, vOnes2))))
vAsstWghts2bookanswer = [6/7, 1/7];                                        // same answer as above 
vAsstWghts2onlyA      = [1, 0];
vAsstWghts2onlyB      = [0, 1];
vAsstWghts2halfAhalfB = [1/2, 1/2];

sTotRisk2 = Math.sqrt(numeric.dot(vAsstWghts2onlyA, numeric.dot(mAsstCov22, vAsstWghts2onlyA)));

// three by three 
vOnes3        = numeric.rep([3],1);
mAsstCor33    = [[1.0, 0.5, 0.2],
                 [0.5, 1.0, 0.3],
                 [0.2, 0.3, 1.0]];
vAsstVol3     = [0.30, 0.20, 0.10];
vAsstWghts3   = [0.5, 0.5, 0.0];
                                                                           // calc covriance from correlations and sigmas  
mVolxXvoly33  = numeric.dot(numeric.transpose([vAsstVol3]), [vAsstVol3]);  // calculate 2x2 matrix of sigma products  sigma_x X sigma_Y
mAsstCov33    = numeric.mul(mVolxXvoly33, mAsstCor33);                     // point-wise multplications for // cov(x,y) = corr(x,y)*vol(x)*vol(y)
mAsstCov33Inv = numeric.inv(mAsstCov33);

vAsstWghts3minvar     = numeric.div(numeric.dot(mAsstCov33Inv, vOnes3),    // minimum variance portfolio = (V^-1 x ones) / (ones' x V^-1 x ones)
                                    numeric.rep([3],numeric.dot(vOnes3,numeric.dot(mAsstCov33Inv, vOnes3))))
vAsstWghts3onlyA      = [1, 0, 0];
vAsstWghts3onlyB      = [0, 1, 0];
vAsstWghts3onlyC      = [0, 0, 1];
vAsstWghts3eqWeigthed = [1/3, 1/3, 1/3];

sTotRisk3 = Math.sqrt(numeric.dot(vAsstWghts3onlyA, numeric.dot(mAsstCov33, vAsstWghts3onlyA)));


function calcPfTest(numAssts)
{
    console.log(numAssts);
    return numAssts*numAssts;
}

function calcPfRiskEnRtn2(mAsstCorr, vAsstVols, vAsstRtns, vAsstWghts, numAssts)  {  // for testing
    console.log("called calcPfRiskEnRtn");
    return { rsk:55, rtn:11 };
}
function calcPfRiskEnRtn(mAsstCorr, vAsstVols, vAsstRtns, vAsstWghts)
{
    numAssts = vAsstWghts.length;
    console.log("calcPfRiskEnRtn:num assets="+numAssts);
    vOnes           = numeric.rep([numAssts],1);
    vAsstWghtsNorm  = numeric.div(vAsstWghts, numeric.sum(vAsstWghts));
                                                                            // calc covriance from correlations and sigmas  
    mVolxXvoly  = numeric.dot(numeric.transpose([vAsstVols]), [vAsstVols]); // calculate 2x2 matrix of sigma products  sigma_x X sigma_Y
    mAsstCov    = numeric.mul(mVolxXvoly, mAsstCorr);                       // point-wise mutplications for // cov(x,y) = corr(x,y)*vol(x)*vol(y)
    mAsstCovInv = numeric.inv(mAsstCov);

    sTotRisk = Math.sqrt(numeric.dot(vAsstWghts, numeric.dot(mAsstCov, vAsstWghts)));
    sTotRtn  = numeric.dot(vAsstWghts, vAsstRtns);

    riskEnReturn = { rsk:sTotRisk/100, rtn:sTotRtn/100 };
    return riskEnReturn;
}

// Matrix-vector product
// b = numeric.dot(mAsstCov, vAsstWghts);
// Matrix inverse.
// Ainv = numeric.inv(A);
// Determinant
// numeric.det(A);
