/* Spherical-code optimiser: push N unit vectors in R^n to max pairwise
 * inner product <= t (default 1/2) by penalty minimisation with restarts.
 *
 *   ./opt n N steps seed [seedfile [nfixed]]
 *
 * Energy  E = sum_{i<j} max(0, g_ij - t)^2 , gradient descent on the sphere
 * with an adaptive step, basin hopping (random kicks) when progress stalls.
 * Prints the best max-inner-product found and writes the best point set. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int n, N;
static double t_target = 0.5;   /* override with env KISS_T */

static unsigned long long rs;
static double urand(void){ rs^=rs<<13; rs^=rs>>7; rs^=rs<<17; return (rs>>11)*(1.0/9007199254740992.0); }
static double nrand(void){ double u=urand()+1e-18,v=urand(); return sqrt(-2*log(u))*cos(6.283185307179586*v); }

static void normalize(double *X){
    for(int i=0;i<N;i++){ double s=0,*x=X+(size_t)i*n;
        for(int k=0;k<n;k++) s+=x[k]*x[k];
        s=1.0/sqrt(s); for(int k=0;k<n;k++) x[k]*=s; }
}

/* energy + gradient; returns E, fills G, sets *mx to the max inner product */
static double engrad(const double *X, double *G, double *mx){
    double E=0, m=-2;
    memset(G,0,sizeof(double)*(size_t)N*n);
    for(int i=0;i<N;i++){
        const double *xi=X+(size_t)i*n; double *gi=G+(size_t)i*n;
        for(int j=i+1;j<N;j++){
            const double *xj=X+(size_t)j*n; double g=0;
            for(int k=0;k<n;k++) g+=xi[k]*xj[k];
            if(g>m) m=g;
            double d=g-t_target;
            if(d>0){ E+=d*d; double c=2*d;
                double *gj=G+(size_t)j*n;
                for(int k=0;k<n;k++){ gi[k]+=c*xj[k]; gj[k]+=c*xi[k]; } }
        }
    }
    *mx=m; return E;
}

static double maxinner(const double *X){
    double m=-2;
    for(int i=0;i<N;i++){ const double *xi=X+(size_t)i*n;
        for(int j=i+1;j<N;j++){ const double *xj=X+(size_t)j*n; double g=0;
            for(int k=0;k<n;k++) g+=xi[k]*xj[k];
            if(g>m) m=g; } }
    return m;
}

int main(int argc,char**argv){
    if(argc<5){ fprintf(stderr,"usage: opt n N steps seed [seedfile [nfixed]]\n"); return 1; }
    { const char*e=getenv("KISS_T"); if(e) t_target=atof(e); }
    n=atoi(argv[1]); N=atoi(argv[2]);
    long steps=atol(argv[3]); rs=strtoull(argv[4],0,10)*2862933555777941757ULL+3037000493ULL;
    int nfixed=0;
    double *X=malloc(sizeof(double)*(size_t)N*n), *G=malloc(sizeof(double)*(size_t)N*n),
           *B=malloc(sizeof(double)*(size_t)N*n), *T=malloc(sizeof(double)*(size_t)N*n);
    for(size_t i=0;i<(size_t)N*n;i++) X[i]=nrand();
    if(argc>5){                       /* seed file: whitespace-separated doubles */
        FILE*f=fopen(argv[5],"r"); if(!f){perror("seed");return 1;}
        size_t cnt=0; while(cnt<(size_t)N*n && fscanf(f,"%lf",&X[cnt])==1) cnt++;
        fclose(f); fprintf(stderr,"seeded %zu of %zu coords\n",cnt,(size_t)N*n);
        if(argc>6) nfixed=atoi(argv[6]);
    }
    normalize(X);
    double best=2, lr=0.02;
    memcpy(B,X,sizeof(double)*(size_t)N*n);
    long stall=0;
    for(long s=0;s<steps;s++){
        double mx,E=engrad(X,G,&mx);
        if(mx<best-1e-12){ best=mx; memcpy(B,X,sizeof(double)*(size_t)N*n); stall=0;
            if(best<=t_target){ fprintf(stderr,"FEASIBLE at step %ld max=%.17g\n",s,best); break; } }
        else stall++;
        if(E==0){ fprintf(stderr,"E=0 at step %ld max=%.17g\n",s,mx); break; }
        /* tangential gradient step */
        for(int i=nfixed;i<N;i++){
            double *x=X+(size_t)i*n,*g=G+(size_t)i*n,d=0;
            for(int k=0;k<n;k++) d+=g[k]*x[k];
            for(int k=0;k<n;k++) x[k]-=lr*(g[k]-d*x[k]);
        }
        normalize(X);
        double mx2=maxinner(X);
        if(mx2<mx) lr*=1.02; else lr*=0.75;
        if(lr<1e-9) lr=1e-9;
        if(stall>4000){                       /* basin hop from the best found */
            memcpy(X,B,sizeof(double)*(size_t)N*n);
            double kick=0.02+0.08*urand();
            for(int i=nfixed;i<N;i++){ double *x=X+(size_t)i*n;
                if(urand()<0.35) for(int k=0;k<n;k++) x[k]+=kick*nrand(); }
            normalize(X); lr=0.02; stall=0;
        }
    }
    printf("n=%d N=%d best_max_inner=%.17g feasible=%d\n",n,N,best,best<=t_target);
    char fn[256]; snprintf(fn,sizeof fn,"%s.out",argv[4]);
    if(argc>5) snprintf(fn,sizeof fn,"%s.n%d.N%d.s%s.out",argv[5],n,N,argv[4]);
    FILE*f=fopen(fn,"w");
    fprintf(f,"# n=%d N=%d max_inner=%.17g\n",n,N,best);
    for(int i=0;i<N;i++){ for(int k=0;k<n;k++) fprintf(f,"%.17g ",B[(size_t)i*n+k]); fputc('\n',f); }
    fclose(f); fprintf(stderr,"wrote %s\n",fn);
    free(X);free(G);free(B);free(T); return 0;
}
