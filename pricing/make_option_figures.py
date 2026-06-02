"""Generate the option-pricing figures for the paper (vector PDF, paper style).
  F1 opt_validation.pdf      - point-delivery Carr-Madan vs Monte Carlo across strikes
  F2 opt_iv_term_structure.pdf - ATM IV term structure: multi-scale vs single-scale CARMA vs market band
  F3 opt_acf_multiscale.pdf  - spot-residual ACF with the 3-component multi-scale fit (slow tail)
"""
import numpy as np, pandas as pd
from scipy.linalg import expm
from scipy.stats import norm, norminvgauss
from scipy.optimize import least_squares, brentq
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
mpl.rcParams.update({"font.size":11,"axes.grid":True,"grid.alpha":0.3,"axes.spines.top":False,
                     "axes.spines.right":False,"figure.dpi":120,"savefig.bbox":"tight"})
from pathlib import Path
ROOT=str(Path(__file__).resolve().parents[1])   # .../Code
FIG=f"{ROOT}/notebooks/figures"

# ---------- model (price factor) ----------
b1_X,b2_X=-1.69107499948807,0.0856060699524467; a1_X,a2_X,a3_X=2.7417368516678,1.85827342924239,0.161163594914844
sigma_T_state=0.76963298150931; mu_Y,sigma_Y=0.004626935939183486,0.9619333921285353
mu_X,delta_X,alpha_X,beta_X=0.032544431425439106,0.3350081939529935,0.9821566545385684,-0.09493955549441858
lam_c=0.0017485173776097822; sigma_J=0.036050456023551204; lsT=lam_c*sigma_T_state
AX=np.array([[0,1,0],[0,0,1],[-a3_X,-a2_X,-a1_X]]); BX=np.array([[0.0],[0.0],[sigma_J]])
cX=np.array([1.0,b1_X,b2_X]); Gamma=np.array([[0.0],[0.0],[lsT]])
ds=pd.read_csv(f"{ROOT}/data/deseasonalised/seasonalities.csv",index_col=0,parse_dates=True)["log_price_seasonal"]; ds.index=pd.to_datetime(ds.index,utc=True)
_lk=(pd.DataFrame({"v":ds.values,"m":ds.index.month,"d":ds.index.dayofweek,"h":ds.index.hour}).groupby(["m","d","h"])["v"].mean().to_dict()); _mn=float(ds.mean())
START=pd.Timestamp("2026-05-13 12:00:00"); END=pd.Timestamp("2026-06-12 19:00:00"); Tpt=(END-START).total_seconds()/3600
def LamS(t): return _lk.get((t.month,t.dayofweek,t.hour),_mn)
def psi_X(th):
    g=np.sqrt(alpha_X**2-beta_X**2); return 1j*mu_X*th+delta_X*(g-np.sqrt(alpha_X**2-(beta_X+1j*th)**2))
def psi_T(th): return 1j*mu_Y*th-0.5*sigma_Y**2*th**2
def kbeta(s,T): return float(BX.flatten()@expm(AX.T*(T-s))@cX)
def kgam(s,T): return float(Gamma.flatten()@expm(AX.T*(T-s))@cX)
def kX(b): return np.real(psi_X(-1j*b));
def kY(g): return np.real(psi_T(-1j*g))
def varG(tau,T,n=1200):
    s=np.linspace(0,tau,n); kb=np.array([kbeta(si,T) for si in s]); kg=np.array([kgam(si,T) for si in s])
    VJ=delta_X*alpha_X**2/(alpha_X**2-beta_X**2)**1.5; return np.trapezoid(VJ*kb**2+sigma_Y**2*kg**2,s)
def a_const(tau,T,n=1200):
    s=np.linspace(tau,T,n); kb=np.array([kbeta(si,T) for si in s]); kg=np.array([kgam(si,T) for si in s])
    return LamS(END)+np.trapezoid(kX(kb),s)+np.trapezoid(kY(kg),s)
def fdc(u,K,lam,a1): Ksh=K+1000.0; z=a1+1j*u; return Ksh*np.exp(-z*(np.log(Ksh)-lam))/((z-1)*z)
def cm(K,tau,T,a1=1.5,M=32768,n_s=1200):
    sG=np.sqrt(max(varG(tau,T),1e-12)); umax=max(250.,12./sG); eta=2*umax/M
    u=(-M//2+np.arange(M))*eta; w=np.full(M,eta); w[0]*=.5; w[-1]*=.5
    s=np.linspace(0,tau,n_s); kb=np.array([kbeta(si,T) for si in s]); kg=np.array([kgam(si,T) for si in s])
    phi=np.exp(np.trapezoid(psi_X(np.outer(u-1j*a1,kb))+psi_T(np.outer(u-1j*a1,kg)),s,axis=1))
    return (np.sum(w*fdc(u,K,a_const(tau,T),a1)*phi)/(2*np.pi)).real
def mc(Ks,tau,T,N=200000,seed=42,n_sub=3):
    dt=1/n_sub; nt=int(round(tau*n_sub)); eA=expm(AX*dt); MXX=(np.linalg.inv(AX)@(eA-np.eye(3))@np.array([[0.],[0.],[1.]])).flatten()/dt
    np.random.seed(seed); Z=np.zeros((N,3)); aJ,bJ=alpha_X*delta_X*dt,beta_X*delta_X*dt
    for _ in range(nt):
        dLY=np.random.normal(mu_Y*dt,sigma_Y*np.sqrt(dt),N); dLJ=norminvgauss.rvs(aJ,bJ,loc=mu_X*dt,scale=delta_X*dt,size=N)
        Z=Z@eA.T+np.outer(lam_c*sigma_T_state*dLY+sigma_J*dLJ,MXX)
    G=Z@(expm(AX*(T-tau)).T@cX); F=np.exp(a_const(tau,T)+G)-1000.0
    return {K:(np.maximum(F-K,0).mean(),np.maximum(F-K,0).std()/np.sqrt(N)) for K in Ks}

# ---------- Black-76 + ACF ----------
def b76(F,K,s,t): sv=s*np.sqrt(t); d1=(np.log(F/K)+.5*sv*sv)/sv; return F*norm.cdf(d1)-K*norm.cdf(d1-sv)
def b76iv(p,F,K,t):
    if p<=max(F-K,0)+1e-10: return np.nan
    try: return brentq(lambda s:b76(F,K,s,t)-p,1e-6,50,maxiter=300)
    except ValueError: return np.nan
pr=pd.read_csv(f"{ROOT}/data/deseasonalised/price_resid.csv",index_col=0)["price_deseasoned"].to_numpy(); pr=pr-pr.mean()
npr=len(pr); v0=pr@pr/npr
def acf(k): return float(pr[:-k]@pr[k:]/((npr-k)*v0))
lags=np.unique(np.round(np.logspace(0,np.log10(2600),45)).astype(int)); rho=np.array([acf(int(k)) for k in lags])
def mdl(th,k): tau=np.exp(th[:3]); return (th[3:][:,None]*np.exp(-k[None,:]/tau[:,None])).sum(0)
sol=least_squares(lambda th:np.concatenate([mdl(th,lags)-rho,[3*(th[3:].sum()-1)]]),
   [np.log(3),np.log(72),np.log(1440),.5,.3,.12],bounds=([np.log(.2),np.log(10),np.log(300),0,0,0],[np.log(10),np.log(300),np.log(4380),1.2,1.2,1.2]))
tau3=np.exp(sol.x[:3]); ww=sol.x[3:]/sol.x[3:].sum(); o=np.argsort(tau3); kap=1/tau3[o]; Vc=ww[o]*v0

# ---------- multi-scale IV term structure ----------
def atm_iv(topt,kap,Vv,N=300000,seed=1,dl=720.,gap=10.):
    T1=topt+gap; T2=T1+dl; u=np.arange(T1,T2+1e-9,1.); wj=np.full(len(u),1.); wj[0]=wj[-1]=.5; wj/=wj.sum()
    a=np.array([LamS(START+pd.Timedelta(hours=float(uj))) for uj in u]).astype(float)
    for i in range(len(kap)): a=a+0.5*Vv[i]*(1-np.exp(-2*kap[i]*(u-topt)))
    rng=np.random.default_rng(seed); Z=np.zeros((N,len(u)))
    for i in range(len(kap)):
        Z+=np.outer(rng.normal(0,np.sqrt(Vv[i]*(1-np.exp(-2*kap[i]*topt))),N),np.exp(-kap[i]*(u-topt)))
    F=np.exp(a[None,:]+Z)@wj-1000.; F0=F.mean(); return b76iv(np.maximum(F-F0,0).mean(),F0,F0,topt/8760.)

# ================= FIGURE 1: validation =================
print("Fig 1: point-delivery CM vs MC ...")
tauv=Tpt-2.0; F0pt=np.exp(a_const(0,Tpt))-1000  # approx fwd level for strike centering
Kline=np.linspace(105,175,40); cmline=np.array([cm(K,tauv,Tpt) for K in Kline])
Kpts=np.array([110,122,130,137,149,160,170.]); mcpts=mc(list(Kpts),tauv,Tpt,N=250000,n_sub=3)
mcv=np.array([mcpts[K][0] for K in Kpts]); mcse=np.array([mcpts[K][1] for K in Kpts])
cmatpts=np.array([cm(K,tauv,Tpt) for K in Kpts]); maxsig=np.max(np.abs(cmatpts-mcv)/mcse)
fig,ax=plt.subplots(figsize=(6.2,4.3))
ax.plot(Kline,cmline,'-',color="#1f4e79",lw=1.8,label="Carr–Madan (Fourier)",zorder=2)
ax.errorbar(Kpts,mcv,yerr=1.96*mcse,fmt='o',ms=5,color="#c0392b",capsize=3,label="Monte Carlo (95% CI)",zorder=3)
ax.set_xlabel("strike $K$ (EUR/MWh)"); ax.set_ylabel("call price on $F(\\tau,T)$ (EUR/MWh)")
ax.set_title("Point-delivery futures call: transform vs simulation")
ax.legend(frameon=False)
ax.text(0.97,0.93,f"max $|C_{{\\rm CM}}-C_{{\\rm MC}}|$ = {maxsig:.1f} MC s.e.",transform=ax.transAxes,
        ha="right",va="top",fontsize=9,color="#555")
fig.savefig(f"{FIG}/opt_validation.pdf"); plt.close(fig)

# ================= FIGURE 2: IV term structure =================
print("Fig 2: IV term structure ...")
mats=[("1w",168.),("2w",350.),("1m",720.),("2m",1440.),("3m",2160.)]
iv3=[atm_iv(t,kap,Vc)*100 for _,t in mats]; iv1=[atm_iv(t,np.array([kap[0]]),np.array([v0]))*100 for _,t in mats]
x=[t/720. for _,t in mats]
fig,ax=plt.subplots(figsize=(6.2,4.3))
ax.axhspan(45,55,color="#7f8c8d",alpha=0.18,label="market O7FM ATM (≈45–55%)")
ax.plot(x,iv3,'o-',color="#1f4e79",lw=1.9,ms=6,label="3-scale CARMA (2.5h/1d/71d)")
ax.plot(x,iv1,'s--',color="#c0392b",lw=1.6,ms=5,label="single-scale CARMA (current fit)")
ax.set_xlabel("option maturity (months)"); ax.set_ylabel("ATM Black-76 implied vol (%)")
ax.set_title("Forward-volatility term structure (monthly base-load)")
ax.set_ylim(-3,62); ax.legend(frameon=False,loc="center right")
ax.set_xticks(x); ax.set_xticklabels([m for m,_ in mats])
fig.savefig(f"{FIG}/opt_iv_term_structure.pdf"); plt.close(fig)

# ================= FIGURE 3: ACF multi-scale =================
print("Fig 3: ACF multi-scale ...")
sol1=least_squares(lambda th:mdl(np.array([th[0],th[0],th[0],th[1],0,0]),lags)-rho,[np.log(3),1.0],
                   bounds=([np.log(.2),0],[np.log(50),1.3]))
fast_only=np.array([sol1.x[0],sol1.x[0],sol1.x[0],sol1.x[1],0,0])
kk=np.logspace(0,np.log10(2600),200)
fig,ax=plt.subplots(figsize=(6.2,4.3))
ax.semilogx(lags,rho,'o',ms=4,color="#c0392b",label="empirical ACF (hourly log-price)",zorder=3)
ax.semilogx(kk,mdl(sol.x,kk),'-',color="#1f4e79",lw=1.9,label="3-scale fit (2.5h/1d/71d)",zorder=2)
ax.semilogx(kk,mdl(fast_only,kk),'--',color="#7f8c8d",lw=1.5,label="single-scale fit (fast only)",zorder=1)
# slow component alone
slow=ww[o][2]*np.exp(-kap[2]*kk)
ax.semilogx(kk,slow,':',color="#27ae60",lw=1.6,label=f"slow component (71d, {ww[o][2]*100:.0f}% var)")
for d,lab in [(24,"1d"),(168,"1w"),(720,"1mo"),(2160,"3mo")]:
    ax.axvline(d,color="#ddd",lw=0.8,zorder=0)
ax.set_xlabel("lag (hours)"); ax.set_ylabel("autocorrelation")
ax.set_title("Spot ACF: the slow tail the single-scale model misses")
ax.set_ylim(-0.02,1.0); ax.legend(frameon=False,fontsize=9)
fig.savefig(f"{FIG}/opt_acf_multiscale.pdf"); plt.close(fig)

print("\nSaved:")
import os
for f in ["opt_validation.pdf","opt_iv_term_structure.pdf","opt_acf_multiscale.pdf"]:
    print(f"  {FIG}/{f}  ({os.path.getsize(FIG+'/'+f)//1024} KB)")
print(f"\nFig2 data: 3-scale IV = {[round(v,1) for v in iv3]}  single = {[round(v,1) for v in iv1]}  (maturities {[m for m,_ in mats]})")
print(f"Fig1: CM vs MC max gap = {maxsig:.2f} s.e.;  Fig3: slow share {ww[o][2]:.2f}, half-life {np.log(2)/kap[2]/24:.0f}d")
