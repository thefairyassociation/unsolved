/* Riesz-energy continuation for spherical codes.
 *
 *   ./riesz n N steps seed [seedfile]
 *
 * Minimises  E_s = sum_{i<j} ||x_i - x_j||^{-s}  on the sphere for a geometrically
 * increasing sequence of exponents s.  As s -> infinity the minimisers approach
 * best-packing configurations, so this homotopy walks a good starting
 * configuration towards one with a small maximum inner product.  This is the
 * scheme (logarithmic / Riesz energy continuation) that produced the dimension-12
 * 841-point code from the classical 840.  The gradient is rescaled by the energy
 * so the step size stays meaningful as s grows. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int n, N;
static unsigned long long rs;
static double urand(void){ rs^=rs<<13; rs^=rs>>7; rs^=rs<<17; return (rs>>11)*(1.0/9007199254740992.0); }
static double nrand(void){ double u=urand()+1e-18,v=urand(); return sqrt(-2*log(u))*cos(6.283185307179586*v); }
static void normalize(double *X){
    for(int i=0;i<N;i++){ double s=0,*x=X+(size_t)i*n;
        for(int k=0;k<n;k++) s+=x[k]*x[k];
        s=1.0/sqrt(s); for(int k=0;k<n;k++) x[k]*=s; }
}
/* E_s and gradient, both divided by E_s; also returns the max inner product.
 * Two passes: the first finds the smallest squared distance so the second can
 * evaluate (r2/r2min)^(-s/2), whose largest term is 1.  Without that rescaling
 * pow() overflows to +inf for large s and the whole configuration turns to NaN
 * (which then reads as a spurious max inner product of -2). */
static double LOGSCALE;   /* log of the rescaling factor, to compare energies across iterations */
static double engrad(const double *X, double *G, double s, double *mx){
    double E=0,m=-2,r2min=1e300;
    for(int i=0;i<N;i++){
        const double *xi=X+(size_t)i*n;
        for(int j=i+1;j<N;j++){
            const double *xj=X+(size_t)j*n; double g=0;
            for(int k=0;k<n;k++) g+=xi[k]*xj[k];
            if(g>m) m=g;
            double r2=2-2*g; if(r2<1e-12) r2=1e-12;
            if(r2<r2min) r2min=r2;
        }
    }
    *mx=m;
    if(!(m>-1.5)) return -1.0;                    /* NaN guard */
    memset(G,0,sizeof(double)*(size_t)N*n);
    for(int i=0;i<N;i++){
        const double *xi=X+(size_t)i*n; double *gi=G+(size_t)i*n;
        for(int j=i+1;j<N;j++){
            const double *xj=X+(size_t)j*n; double g=0;
            for(int k=0;k<n;k++) g+=xi[k]*xj[k];
            double r2=2-2*g; if(r2<1e-12) r2=1e-12;
            double e=pow(r2/r2min,-s/2); E+=e;
            double c=s*e/r2;
            double *gj=G+(size_t)j*n;
            for(int k=0;k<n;k++){ double dk=xi[k]-xj[k]; gi[k]-=c*dk; gj[k]+=c*dk; }
        }
    }
    if(E>0) for(size_t t=0;t<(size_t)N*n;t++) G[t]/=E;
    LOGSCALE = log(E) - (s / 2) * log(r2min);   /* = log sum r^-s, scale invariant */
    return E;
}
/* log sum_{i<j} ||xi-xj||^{-s}, computed in a rescaled way so it never overflows */
static double logenergy(const double *X, double s){
    double E=0,r2min=1e300;
    for(int i=0;i<N;i++){ const double *xi=X+(size_t)i*n;
        for(int j=i+1;j<N;j++){ const double *xj=X+(size_t)j*n; double g=0;
            for(int k=0;k<n;k++) g+=xi[k]*xj[k];
            double r2=2-2*g; if(r2<1e-12) r2=1e-12;
            if(r2<r2min) r2min=r2; } }
    for(int i=0;i<N;i++){ const double *xi=X+(size_t)i*n;
        for(int j=i+1;j<N;j++){ const double *xj=X+(size_t)j*n; double g=0;
            for(int k=0;k<n;k++) g+=xi[k]*xj[k];
            double r2=2-2*g; if(r2<1e-12) r2=1e-12;
            E+=pow(r2/r2min,-s/2); } }
    if(!(E>0)) return 1e300;
    return log(E) - (s/2)*log(r2min);
}
static double maxinner(const double*X){
    double m=-2;
    for(int i=0;i<N;i++){ const double*xi=X+(size_t)i*n;
        for(int j=i+1;j<N;j++){ const double*xj=X+(size_t)j*n; double g=0;
            for(int k=0;k<n;k++) g+=xi[k]*xj[k];
            if(g>m) m=g; } }
    return m;
}
int main(int argc,char**argv){
    if(argc<5){ fprintf(stderr,"usage: riesz n N steps seed [seedfile]\n"); return 1; }
    n=atoi(argv[1]); N=atoi(argv[2]); long steps=atol(argv[3]);
    rs=strtoull(argv[4],0,10)*2862933555777941757ULL+3037000493ULL;
    for(int i=0;i<40;i++) urand();
    double *X=malloc(sizeof(double)*(size_t)N*n),*G=malloc(sizeof(double)*(size_t)N*n),
           *B=malloc(sizeof(double)*(size_t)N*n);
    for(size_t i=0;i<(size_t)N*n;i++) X[i]=nrand();
    if(argc>5){ FILE*f=fopen(argv[5],"r"); if(!f){perror("seed");return 1;}
        size_t c=0; while(c<(size_t)N*n && fscanf(f,"%lf",&X[c])==1) c++; fclose(f); }
    normalize(X);
    double best=maxinner(X); memcpy(B,X,sizeof(double)*(size_t)N*n);
    /* continuation schedule: start near the logarithmic energy (s -> 0) and
     * raise the exponent slowly; KISS_S0/KISS_SMUL/KISS_SMAX override it. */
    double s0=0.25, smul=1.12, smax=60000.0;
    { const char*e;
      if((e=getenv("KISS_S0"))) s0=atof(e);
      if((e=getenv("KISS_SMUL"))) smul=atof(e);
      if((e=getenv("KISS_SMAX"))) smax=atof(e); }
    int nstage=1; for(double q=s0;q<smax;q*=smul) nstage++;
    long per = steps/nstage > 40 ? steps/nstage : 40;
    for(double s=s0; s<smax; s*=smul){
        double lr=0.05;
        double *Y=malloc(sizeof(double)*(size_t)N*n);
        for(long it=0; it<per; it++){
            double mx,E=engrad(X,G,s,&mx);
            if(E<0){ fprintf(stderr,"numerical breakdown at s=%.0f\n",s); free(Y); goto done; }
            if(mx<best-1e-13){ best=mx; memcpy(B,X,sizeof(double)*(size_t)N*n);
                if(best<=0.5+1e-13){ fprintf(stderr,"FEASIBLE s=%.0f max=%.17g\n",s,best); free(Y); goto done; } }
            double E0=LOGSCALE;
            /* tangential descent direction */
            for(int i=0;i<N;i++){ double *x=X+(size_t)i*n,*g=G+(size_t)i*n,d=0;
                for(int k=0;k<n;k++) d+=g[k]*x[k];
                for(int k=0;k<n;k++) g[k]-=d*x[k]; }
            int accepted=0;
            for(int back=0; back<24; back++){
                memcpy(Y,X,sizeof(double)*(size_t)N*n);
                for(size_t t=0;t<(size_t)N*n;t++) Y[t]-=lr*G[t];
                for(int i=0;i<N;i++){ double q=0,*y=Y+(size_t)i*n;
                    for(int k=0;k<n;k++) q+=y[k]*y[k];
                    q=1.0/sqrt(q); for(int k=0;k<n;k++) y[k]*=q; }
                if(logenergy(Y,s) < E0){ memcpy(X,Y,sizeof(double)*(size_t)N*n); accepted=1; break; }
                lr*=0.5;
            }
            if(!accepted){ lr=0.05; break; }      /* stage converged */
            lr*=1.6; if(lr>0.5) lr=0.5;
        }
        free(Y);
    }
done:
    printf("n=%d N=%d best_max_inner=%.17g feasible=%d\n",n,N,best,best<=0.5+1e-12);
    char fn[512];
    if(argc>5) snprintf(fn,sizeof fn,"%s.riesz.s%s.out",argv[5],argv[4]);
    else snprintf(fn,sizeof fn,"riesz_%s.out",argv[4]);
    FILE*f=fopen(fn,"w");
    fprintf(f,"# n=%d N=%d max_inner=%.17g\n",n,N,best);
    for(int i=0;i<N;i++){ for(int k=0;k<n;k++) fprintf(f,"%.17g ",B[(size_t)i*n+k]); fputc('\n',f); }
    fclose(f); return 0;
}
