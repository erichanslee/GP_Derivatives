// y = quadraticform(A, x)
#include "mex.h"
#include <math.h>

/*
Mx_in: Vector of length n(deg+1)^d
row_in: Vector of length n(deg+1)^d
ku_in: Vector of length sum(mi)

Mx_in are the non-zero elements of the n-by-prod(mi) interpolation matrix. Each row
has (deg+1) non-zeros and these values are contigous chunks.
*/

/* Input Arguments */
#define Mx_in prhs[0]
#define inds_in prhs[1]
#define ku_in prhs[2]
#define dim_in prhs[3]
#define deg_in prhs[4]

/* Output Arguments */
#define diag plhs[0]

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    mwSize mMx, nMx, n, mInds, nInds, mKu, nKu, m, ndim, d;
    double *Mx, *Ku ;
    int *inds, *dim, *deg, npt;
    deg = (int *) mxGetData(deg_in);

    if (nrhs != 5) {
        mexErrMsgTxt("Five input arguments required.");
    }
    else if (nlhs > 1) {
        mexErrMsgTxt("Too many output arguments.");
    }

    // Check size input
    d = mxGetM(dim_in);
    ndim = mxGetN(dim_in);
    if (ndim != 1) {
        mexErrMsgTxt("Dim must have size d x 1");   
    }
    npt = (int) (pow(*deg+1,d)+0.1);
    
    // Check Mx input
    mMx = mxGetM(Mx_in);
    nMx = mxGetN(Mx_in);
    if (mMx % npt != 0 || nMx != 1) {
        mexErrMsgTxt("Mx must have size (deg+1)^d*npt x 1");
    }
    n = mMx/npt;
    
    // Check inds input
    mInds = mxGetM(inds_in);
    nInds = mxGetN(inds_in);
    if (mInds != mMx || nInds != nMx) {
        mexErrMsgTxt("First and second argument must have the same size");
    }

    // Check inds input
    mKu = mxGetM(ku_in);
    nKu = mxGetN(ku_in);
    if (nKu != 1) {
        mexErrMsgTxt("Third argument must be a column vector");
    }
    m = mKu;

    Mx = mxGetPr(Mx_in);
    inds = (int *) mxGetData(inds_in);
    Ku = mxGetPr(ku_in);
    dim = (int *) mxGetData(dim_in);

    // Compute quadratic forms
    diag = mxCreateNumericMatrix(n, 1, mxDOUBLE_CLASS, mxREAL);
    double *q, y, *K, prod;
    int *ind, indj;
    q = (double *) malloc(npt*sizeof(double));
    ind = (int *) malloc(d*npt*sizeof(int));
    int i, j, k, l;
    for (i = 0; i < n; ++i) {

        // Extract q and row indices
        for(j = 0; j < npt; j++) {
            q[j] = Mx[j+npt*i];
            indj = inds[j+npt*i];
            for (k = d-1; k >= 0; k--){
                ind[k+j*d] = indj%dim[k];
                indj = (indj-ind[k+j*d])/dim[k];
                ind[k+j*d]--;
            }
        }
        
        // Compute the quadratic form
        y = 0.0;
        for(j = 0; j < npt; j++) {
            for(k = 0; k < npt; k++) {
                prod = q[j]*q[k];
                K = Ku;
                for (l = 0; l < d; l++) {
                    prod *= K[abs(ind[j*d+l]-ind[k*d+l])];
                    K += dim[l];
                }
                y += prod;
            }
        }

        // Save to output array
        ((double*)mxGetPr(diag))[i] = y;
    }
}
