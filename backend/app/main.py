from fastapi import FastAPI,Request
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from collections import defaultdict,deque
import hashlib,time,shutil
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from .config import CORS_ORIGINS,PROJECT_DIR,JWT_SECRET,MODEL_PATH,BASE_DIR
from .database import init_db,db
from .routers import auth,prediction,analysis,history,optimization,developer,training
@asynccontextmanager
async def lifespan(app):
    startup()
    yield
app=FastAPI(title='CAPON API — MgO CO₂ Adsorption Platform',version='1.0.0',description='Research-screening predictions with study-grouped validation. Capacity target stays in mg/g; RMSE is also reported in mmol/g.',lifespan=lifespan)
app.add_middleware(CORSMiddleware,allow_origins=[x.strip() for x in CORS_ORIGINS if x.strip()],allow_credentials=True,allow_methods=['GET','POST','PATCH','DELETE','OPTIONS'],allow_headers=['Authorization','Content-Type'])
_rate_buckets=defaultdict(deque)
@app.middleware('http')
async def rate_limit(request:Request,call_next):
    ip=request.client.host if request.client else 'unknown'; now=time.time(); is_auth=request.url.path.startswith('/auth/') and request.method=='POST'; key=(ip,'auth' if is_auth else 'general'); window=300 if is_auth else 60; limit=20 if is_auth else 240
    bucket=_rate_buckets[key]
    while bucket and now-bucket[0]>window: bucket.popleft()
    if len(bucket)>=limit: return JSONResponse(status_code=429,content={'detail':'Rate limit exceeded; try again later.'})
    bucket.append(now)
    return await call_next(request)
def startup():
    if len(JWT_SECRET)<32: raise RuntimeError('Set JWT_SECRET to a random secret of at least 32 characters in the environment before startup.')
    init_db()
    # Record the bundled normalized source dataset once; it remains versioned and auditable.
    p=PROJECT_DIR/'data/mgo_database_source_normalized.csv'
    if p.exists():
        digest=hashlib.sha256(p.read_bytes()).hexdigest()
        with db() as c:
            if not c.execute('SELECT 1 FROM dataset_versions LIMIT 1').fetchone():
                import pandas as pd
                count=len(pd.read_csv(p,header=None,engine='python'))
                version='dataset-'+digest[:12]
                c.execute('INSERT INTO dataset_versions(version,filename,sha256,row_count,notes) VALUES(?,?,?,?,?)',(version,p.name,digest,count,'Bundled normalized input; source CSV repair details in dataset_inspection.json'))
                snapshot=PROJECT_DIR/'data/versions'/f'{version}.csv'; snapshot.parent.mkdir(parents=True,exist_ok=True)
                if not snapshot.exists(): shutil.copy2(p,snapshot)
    if MODEL_PATH.exists():
        try:
            import joblib,json
            artifact=joblib.load(MODEL_PATH); version=artifact.get('version','unknown')
            report=json.loads((BASE_DIR/'reports/evaluation.json').read_text())
            metrics=report['validation']['study_grouped_nested_5_fold'][report['model']['name']]['metrics']
            with db() as c:
                c.execute("INSERT OR IGNORE INTO model_versions(version,artifact_path,metrics_json,status) VALUES(?,?,?,'deployed')",(version,str(MODEL_PATH),json.dumps(metrics)))
                if not c.execute("SELECT 1 FROM deployments WHERE model_version=? AND action='initial'",(version,)).fetchone(): c.execute("INSERT INTO deployments(model_version,action) VALUES(?,'initial')",(version,))
        except Exception: pass
@app.middleware('http')
async def security_headers(request:Request,call_next):
    response=await call_next(request)
    response.headers['X-Content-Type-Options']='nosniff'; response.headers['X-Frame-Options']='DENY'; response.headers['Referrer-Policy']='strict-origin-when-cross-origin'
    return response
@app.get('/health')
def health():
    from .services.prediction_service import runtime
    return {'status':'ok','model_loaded':bool(runtime.payload),'model_error':runtime.error}
for router in (auth.router,prediction.router,analysis.router,history.router,optimization.router,developer.router,training.router): app.include_router(router)
@app.exception_handler(Exception)
async def unexpected_error(request,exc):
    return JSONResponse(status_code=500,content={'detail':'Internal server error'})
