import React, { useState, useEffect, useRef, useCallback, useMemo } from "react";
import * as THREE from "three";

/* =====================================================================
   CUBOID INSERTION CHECKER
   ---------------------------------------------------------------------
   Section 1: GEOMETRY + SOLVER MODULE  (UI-independent, pure functions)
   Section 2: 3D VIEWPORT (three.js)
   Section 3: APPLICATION UI
   All lengths in millimetres. Rotations are quaternions [x,y,z,w].
   ===================================================================== */

/* ============================ SECTION 1 ============================ */
/* ------ typed data models (JSDoc; extract to .ts when porting) ----- */
/**
 * @typedef {{x:number,y:number,z:number,wall:number,face:string}} BoxDimensions
 * @typedef {{w:number,h:number,offU:number,offV:number,radius:number,clearance:number}} OpeningDimensions
 * @typedef {{l:number,w:number,h:number,clearance:number,finalMode:string,fx:number,fy:number,fz:number,anyOrientation:boolean}} CuboidDimensions
 * @typedef {{p:[number,number,number], q:[number,number,number,number]}} Pose
 * @typedef {{maxTimeMs:number,stepMm:number,rrtStep:number,maxNodes:number}} SolverSettings
 * @typedef {{status:string,path:Pose[]|null,stats:object,messages:string[]}} SolverResult
 */

const FACES = {
  front:  { axis: 2, sign: +1, u: 0, v: 1, label: "Front (+Z)" },
  back:   { axis: 2, sign: -1, u: 0, v: 1, label: "Back (−Z)" },
  right:  { axis: 0, sign: +1, u: 2, v: 1, label: "Right (+X)" },
  left:   { axis: 0, sign: -1, u: 2, v: 1, label: "Left (−X)" },
  top:    { axis: 1, sign: +1, u: 0, v: 2, label: "Top (+Y)" },
  bottom: { axis: 1, sign: -1, u: 0, v: 2, label: "Bottom (−Y)" },
};

const vAdd=(a,b)=>[a[0]+b[0],a[1]+b[1],a[2]+b[2]];
const vSub=(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]];
const vScale=(a,s)=>[a[0]*s,a[1]*s,a[2]*s];
const vDot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const vCross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const vLen=(a)=>Math.hypot(a[0],a[1],a[2]);
const vLerp=(a,b,t)=>[a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t,a[2]+(b[2]-a[2])*t];

/* ----------------------------- quaternions ----------------------------- */
const Q_ID = [0,0,0,1];
function qNorm(q){const l=Math.hypot(q[0],q[1],q[2],q[3])||1;return[q[0]/l,q[1]/l,q[2]/l,q[3]/l];}
function qMul(a,b){
  return qNorm([
    a[3]*b[0]+a[0]*b[3]+a[1]*b[2]-a[2]*b[1],
    a[3]*b[1]-a[0]*b[2]+a[1]*b[3]+a[2]*b[0],
    a[3]*b[2]+a[0]*b[1]-a[1]*b[0]+a[2]*b[3],
    a[3]*b[3]-a[0]*b[0]-a[1]*b[1]-a[2]*b[2],
  ]);
}
function qAxisAngle(axis,ang){const s=Math.sin(ang/2),l=vLen(axis)||1;return qNorm([axis[0]/l*s,axis[1]/l*s,axis[2]/l*s,Math.cos(ang/2)]);}
function qAngle(a,b){const d=Math.min(1,Math.abs(a[0]*b[0]+a[1]*b[1]+a[2]*b[2]+a[3]*b[3]));return 2*Math.acos(d);}
function qSlerp(a,b,t){
  let d=a[0]*b[0]+a[1]*b[1]+a[2]*b[2]+a[3]*b[3];
  let bb=b;
  if(d<0){bb=[-b[0],-b[1],-b[2],-b[3]];d=-d;}
  if(d>0.9995){return qNorm([a[0]+(bb[0]-a[0])*t,a[1]+(bb[1]-a[1])*t,a[2]+(bb[2]-a[2])*t,a[3]+(bb[3]-a[3])*t]);}
  const th=Math.acos(d), s=Math.sin(th);
  const wa=Math.sin((1-t)*th)/s, wb=Math.sin(t*th)/s;
  return qNorm([a[0]*wa+bb[0]*wb,a[1]*wa+bb[1]*wb,a[2]*wa+bb[2]*wb,a[3]*wa+bb[3]*wb]);
}
function qRandom(rng){ // Shoemake uniform random rotation
  const u1=rng(),u2=rng(),u3=rng();
  const s1=Math.sqrt(1-u1), s2=Math.sqrt(u1);
  return [s1*Math.sin(2*Math.PI*u2), s1*Math.cos(2*Math.PI*u2), s2*Math.sin(2*Math.PI*u3), s2*Math.cos(2*Math.PI*u3)];
}
function qToAxes(q){ // returns object local axes as world vectors (columns of R)
  const [x,y,z,w]=q;
  return [
    [1-2*(y*y+z*z), 2*(x*y+z*w),   2*(x*z-y*w)],
    [2*(x*y-z*w),   1-2*(x*x+z*z), 2*(y*z+x*w)],
    [2*(x*z+y*w),   2*(y*z-x*w),   1-2*(x*x+y*y)],
  ];
}

/* the 24 axis-aligned orientations of a cuboid (rotation group of the cube) */
function axisAlignedQuats(){
  const quats=[]; const seen=new Set();
  const base=[Q_ID,
    qAxisAngle([1,0,0],Math.PI/2),qAxisAngle([1,0,0],Math.PI),qAxisAngle([1,0,0],-Math.PI/2),
    qAxisAngle([0,0,1],Math.PI/2),qAxisAngle([0,0,1],-Math.PI/2)];
  for(const b of base){
    for(let k=0;k<4;k++){
      const q=qMul(qAxisAngle([0,1,0],k*Math.PI/2),b);
      const axes=qToAxes(q);
      const key=axes.flat().map(v=>Math.round(v)).join(",");
      if(!seen.has(key)){seen.add(key);quats.push(q);}
    }
  }
  return quats;
}
const AA_QUATS = axisAlignedQuats();

/* --------------------- enclosure wall panel model --------------------- */
/** usable opening after clearance + conservative corner-radius shrink */
function usableOpening(op){
  const rShrink = op.radius>0 ? 2*op.radius*(1-Math.SQRT1_2) : 0; // conservative rect inside rounded rect
  return {
    w: op.w - 2*op.clearance - rShrink,
    h: op.h - 2*op.clearance - rShrink,
    offU: op.offU, offV: op.offV,
  };
}

/** Build the 6 walls as axis-aligned panels; opening wall becomes 4 strips. */
function buildPanels(box, opening, faceKey){
  const f=FACES[faceKey];
  const dims=[box.x,box.y,box.z];
  const t=box.wall;
  const panels=[];
  const uo=usableOpening(opening);
  for(const key of Object.keys(FACES)){
    const g=FACES[key];
    const c=[0,0,0], h=[0,0,0];
    c[g.axis]=g.sign*(dims[g.axis]/2+t/2);
    h[g.axis]=t/2;
    h[g.u]=dims[g.u]/2+t;
    h[g.v]=dims[g.v]/2+t;
    if(key!==faceKey){
      panels.push({center:c,half:h,isOpeningWall:false,name:key});
      continue;
    }
    // opening wall -> 4 strips around usable opening rect (in u,v plane)
    const Hu=dims[g.u]/2+t, Hv=dims[g.v]/2+t;
    const ow2=Math.max(0,uo.w/2), oh2=Math.max(0,uo.h/2);
    const ou=uo.offU, ov=uo.offV;
    const strips=[
      {u0:-Hu, u1:ou-ow2, v0:-Hv, v1:Hv},          // left of hole
      {u0:ou+ow2, u1:Hu,  v0:-Hv, v1:Hv},          // right of hole
      {u0:ou-ow2, u1:ou+ow2, v0:-Hv, v1:ov-oh2},   // below hole
      {u0:ou-ow2, u1:ou+ow2, v0:ov+oh2, v1:Hv},    // above hole
    ];
    for(const s of strips){
      if(s.u1-s.u0<=1e-9||s.v1-s.v0<=1e-9) continue;
      const pc=[0,0,0],ph=[0,0,0];
      pc[g.axis]=c[g.axis]; ph[g.axis]=t/2;
      pc[g.u]=(s.u0+s.u1)/2; ph[g.u]=(s.u1-s.u0)/2;
      pc[g.v]=(s.v0+s.v1)/2; ph[g.v]=(s.v1-s.v0)/2;
      panels.push({center:pc,half:ph,isOpeningWall:true,name:faceKey+"-strip"});
    }
  }
  return panels;
}

/* --------------------- OBB vs AABB separation (SAT) --------------------- */
/** Returns the best separation gap (>0 = separated by at least gap along some axis;
 *  <=0 = considered colliding). Object: center ca, half extents ha, quaternion q.
 *  Panel: axis-aligned, center cb, half hb. */
function obbPanelGap(ca,ha,axes,cb,hb){
  const d=vSub(cb,ca);
  let best=-Infinity;
  // world axes (panel face normals)
  for(let i=0;i<3;i++){
    const projA=ha[0]*Math.abs(axes[0][i])+ha[1]*Math.abs(axes[1][i])+ha[2]*Math.abs(axes[2][i]);
    const gap=Math.abs(d[i])-projA-hb[i];
    if(gap>best)best=gap;
  }
  // object axes
  for(let i=0;i<3;i++){
    const L=axes[i];
    const projB=hb[0]*Math.abs(L[0])+hb[1]*Math.abs(L[1])+hb[2]*Math.abs(L[2]);
    const gap=Math.abs(vDot(d,L))-ha[i]-projB;
    if(gap>best)best=gap;
  }
  // cross products
  for(let i=0;i<3;i++){
    for(let j=0;j<3;j++){
      const e=[0,0,0];e[j]=1;
      const L=vCross(axes[i],e);
      const len=vLen(L);
      if(len<1e-8)continue;
      const Ln=vScale(L,1/len);
      const projA=ha[0]*Math.abs(vDot(axes[0],Ln))+ha[1]*Math.abs(vDot(axes[1],Ln))+ha[2]*Math.abs(vDot(axes[2],Ln));
      const projB=hb[0]*Math.abs(Ln[0])+hb[1]*Math.abs(Ln[1])+hb[2]*Math.abs(Ln[2]);
      const gap=Math.abs(vDot(d,Ln))-projA-projB;
      if(gap>best)best=gap;
    }
  }
  return best;
}

/* ----------------------------- pose checks ----------------------------- */
function poseGapInfo(pose,half,panels,counter){
  if(counter)counter.n++;
  const axes=qToAxes(pose.q);
  let minAll=Infinity,minOpen=Infinity,minWall=Infinity,collide=false;
  for(const p of panels){
    const g=obbPanelGap(pose.p,half,axes,p.center,p.half);
    if(g<=0)collide=true;
    if(g<minAll)minAll=g;
    if(p.isOpeningWall){if(g<minOpen)minOpen=g;}else{if(g<minWall)minWall=g;}
  }
  return {collide,minAll,minOpen,minWall};
}
function poseFree(pose,half,panels,counter){
  if(counter)counter.n++;
  const axes=qToAxes(pose.q);
  for(const p of panels){
    if(obbPanelGap(pose.p,half,axes,p.center,p.half)<=0)return false;
  }
  return true;
}
function edgeFree(a,b,half,panels,stepMm,rotRadius,counter){
  const dist=vLen(vSub(b.p,a.p))+qAngle(a.q,b.q)*rotRadius;
  const n=Math.max(2,Math.ceil(dist/stepMm)+1);
  for(let i=0;i<=n;i++){
    const t=i/n;
    const pose={p:vLerp(a.p,b.p,t),q:qSlerp(a.q,b.q,t)};
    if(!poseFree(pose,half,panels,counter))return false;
  }
  return true;
}

/* ------------------------------ validation ------------------------------ */
function validateConfig(cfg){
  const errs=[];
  const {box,opening,obj}=cfg;
  const pos=(v,name)=>{if(!(v>0))errs.push(name+" must be a positive number.");};
  pos(box.x,"Internal width X");pos(box.y,"Internal height Y");pos(box.z,"Internal depth Z");
  pos(box.wall,"Wall thickness");pos(opening.w,"Opening width");pos(opening.h,"Opening height");
  pos(obj.l,"Object length");pos(obj.w,"Object width");pos(obj.h,"Object height");
  if(opening.radius<0)errs.push("Corner radius cannot be negative.");
  if(opening.clearance<0||obj.clearance<0)errs.push("Clearances cannot be negative.");
  if(opening.radius*2>Math.min(opening.w,opening.h))errs.push("Corner radius is larger than half the opening size.");
  const f=FACES[box.face];
  if(f){
    const dims=[box.x,box.y,box.z];
    const fu=dims[f.u]/2, fv=dims[f.v]/2;
    if(Math.abs(opening.offU)+opening.w/2>fu+1e-9)
      errs.push("Opening extends beyond the selected face horizontally (max half-width "+fu.toFixed(1)+" mm).");
    if(Math.abs(opening.offV)+opening.h/2>fv+1e-9)
      errs.push("Opening extends beyond the selected face vertically (max half-height "+fv.toFixed(1)+" mm).");
  }
  const uo=usableOpening(opening);
  if(uo.w<=0||uo.h<=0)errs.push("Usable opening is zero or negative after clearance / corner-radius reduction.");
  if(obj.finalMode==="manual"){
    const hx=[obj.l/2,obj.w/2,obj.h/2]; // rough: axis-aligned bound check only
    if(Math.abs(obj.fx)>box.x/2||Math.abs(obj.fy)>box.y/2||Math.abs(obj.fz)>box.z/2)
      errs.push("Manual final position centre lies outside the box interior.");
  }
  return errs;
}

/* -------------------------- preliminary checks -------------------------- */
function perms3(d){
  const [a,b,c]=d;
  return [[a,b,c],[a,c,b],[b,a,c],[b,c,a],[c,a,b],[c,b,a]];
}
function prelimChecks(cfg){
  const {box,opening,obj}=cfg;
  const uo=usableOpening(opening);
  const clr=obj.clearance;
  const dims=[obj.l,obj.w,obj.h];
  const eff=dims.map(d=>d+2*clr);              // effective object incl. tolerance
  const interior=[box.x,box.y,box.z];
  const f=FACES[box.face];
  const openW=uo.w, openH=uo.h;
  const sortedEff=[...eff].sort((a,b)=>a-b);
  const sortedInt=[...interior].sort((a,b)=>a-b);

  // 1. direct axis-aligned insertion
  let direct=false, directOrient=null;
  for(const p of perms3(eff)){
    // p = object extents mapped to [u, v, axis(normal)] of the opening face
    const fitsOpening=(p[0]<=openW&&p[1]<=openH);
    const world=[0,0,0];
    world[f.u]=p[0];world[f.v]=p[1];world[f.axis]=p[2];
    const fitsInterior=world[0]<=interior[0]&&world[1]<=interior[1]&&world[2]<=interior[2];
    // opening offset must let the cross-section reach a valid interior lateral position
    const roomU=interior[f.u]/2-p[0]/2, roomV=interior[f.v]/2-p[1]/2;
    const reachable=Math.abs(uo.offU)<=roomU+1e-9&&Math.abs(uo.offV)<=roomV+1e-9;
    if(fitsOpening&&fitsInterior&&reachable){direct=true;directOrient=p;break;}
  }

  // 2. opening feasibility (necessary condition: any support width >= min dim)
  const minDim=sortedEff[0];
  const openingPossible=minDim<=Math.min(openW,openH);
  // any axis-aligned cross-section fits?
  let crossFits=false;
  for(const p of perms3(eff)){if(p[0]<=openW&&p[1]<=openH){crossFits=true;break;}}

  // 3. final containment
  let containAA=false;
  for(const p of perms3(eff)){if(p[0]<=interior[0]&&p[1]<=interior[1]&&p[2]<=interior[2]){containAA=true;break;}}
  const spaceDiag=Math.hypot(...interior);
  const containNecessary=sortedEff[0]<=sortedInt[0]&&sortedEff[2]<=spaceDiag;
  const containPossible=containAA||( !obj.anyOrientation ? false : containNecessary );

  const provablyImpossible=[];
  if(!openingPossible)provablyImpossible.push(
    "The smallest object dimension ("+sortedEff[0].toFixed(1)+" mm incl. tolerance) exceeds the smaller usable opening side ("+Math.min(openW,openH).toFixed(1)+" mm). No orientation can pass.");
  if(!containNecessary)provablyImpossible.push(
    sortedEff[0]>sortedInt[0]
      ?"The smallest object dimension exceeds the smallest interior dimension; the object cannot fit inside in any orientation."
      :"The longest object dimension ("+sortedEff[2].toFixed(1)+" mm) exceeds the interior space diagonal ("+spaceDiag.toFixed(1)+" mm).");
  if(!obj.anyOrientation&&!containAA&&obj.finalMode!=="manual")
    provablyImpossible.push("No axis-aligned orientation of the object fits inside the interior, and the final orientation is restricted to axis-aligned.");

  return {
    direct, directOrient, openingPossible, crossFits, containAA, containPossible, containNecessary,
    provablyImpossible,
    usable:{openW,openH},
    effDims:eff,
  };
}

/* ------------------------------ full solver ------------------------------ */
function mulberry32(seed){let a=seed>>>0;return function(){a|=0;a=(a+0x6D2B79F5)|0;let t=Math.imul(a^(a>>>15),1|a);t=(t+Math.imul(t^(t>>>7),61|t))^t;return((t^(t>>>14))>>>0)/4294967296;};}
const sleep=(ms)=>new Promise(r=>setTimeout(r,ms));

function goalPosition(cfg){
  const {obj}=cfg;
  return obj.finalMode==="manual"?[obj.fx,obj.fy,obj.fz]:[0,0,0];
}
function goalQuats(cfg){
  return cfg.obj.anyOrientation?AA_QUATS:[Q_ID];
}
function startPose(cfg,q,depthOut){
  const f=FACES[cfg.box.face];
  const uo=usableOpening(cfg.opening);
  const p=[0,0,0];
  const dims=[cfg.box.x,cfg.box.y,cfg.box.z];
  p[f.u]=uo.offU; p[f.v]=uo.offV;
  p[f.axis]=f.sign*(dims[f.axis]/2+cfg.box.wall+depthOut);
  return {p,q};
}

/** Densify a coarse pose path into evenly spaced frames. */
function densifyPath(path,half,stepMm){
  const rotR=vLen(half);
  const out=[path[0]];
  for(let i=1;i<path.length;i++){
    const a=path[i-1],b=path[i];
    const d=vLen(vSub(b.p,a.p))+qAngle(a.q,b.q)*rotR;
    const n=Math.max(1,Math.ceil(d/stepMm));
    for(let k=1;k<=n;k++){
      out.push({p:vLerp(a.p,b.p,k/n),q:qSlerp(a.q,b.q,k/n)});
    }
  }
  return out;
}

async function solveInsertion(cfg,settings,onProgress,isExtraction=false){
  const t0=performance.now();
  const counter={n:0};
  const {box,opening,obj}=cfg;
  const panels=buildPanels(box,opening,box.face);
  const half=[obj.l/2+obj.clearance,obj.w/2+obj.clearance,obj.h/2+obj.clearance];
  const rotR=vLen(half);
  const f=FACES[box.face];
  const dims=[box.x,box.y,box.z];
  const outDepth=Math.max(...[obj.l,obj.w,obj.h])/2+box.wall+20;
  const gPos=goalPosition(cfg);
  const gQuats=goalQuats(cfg);
  const messages=[];
  const stepMm=settings.stepMm;
  const timeUp=()=>performance.now()-t0>settings.maxTimeMs;

  if(isExtraction){
    // Extraction: reverse the problem
    // Start: object inside at final position (gPos, random/custom quat)
    // Goal: object outside (reversed start position)
    const startQuats=cfg.obj.anyOrientation?AA_QUATS:[Q_ID];
    const startPoses=startQuats.map(q=>({p:gPos,q})).filter(pose=>poseFree(pose,half,panels,counter));
    if(startPoses.length===0){
      return {status:"no_goal",path:null,messages:["No collision-free start pose exists at the specified interior position."],
        stats:{timeMs:performance.now()-t0,evaluated:counter.n,stage:"goal check"}};
    }
    const outsidePos=[0,0,0];
    outsidePos[f.axis]=f.sign*(dims[f.axis]/2+box.wall+outDepth);
    const goals=gQuats.map(q=>({p:outsidePos,q})).filter(pose=>poseFree(pose,half,panels,counter));
    if(goals.length===0){
      return {status:"no_goal",path:null,messages:["No collision-free external pose exists."],
        stats:{timeMs:performance.now()-t0,evaluated:counter.n,stage:"goal check"}};
    }
    messages.push("EXTRACTION MODE: pulling object out from inside the enclosure.");
    const finish=(rawPath,stage)=>{
      const path=densifyPath(rawPath,half,Math.min(stepMm,3));
      let tight={minAll:Infinity,minOpen:Infinity,minWall:Infinity,idx:0};
      path.forEach((pose,i)=>{
        const g=poseGapInfo(pose,half,panels,counter);
        if(g.minAll<tight.minAll)tight={...g,idx:i};
        tight.minOpen=Math.min(tight.minOpen,g.minOpen);
        tight.minWall=Math.min(tight.minWall,g.minWall);
      });
      return {status:"feasible",path,messages,
        stats:{timeMs:performance.now()-t0,evaluated:counter.n,stage,
          minClearance:tight.minAll,minOpening:tight.minOpen,minWall:tight.minWall,
          tightIdx:tight.idx,tightPose:path[tight.idx],finalPose:path[path.length-1]}};
    };
    // Try direct extraction (reverse)
    for(const start of startPoses){
      for(const goal of goals){
        if(timeUp())break;
        if(edgeFree(start,goal,half,panels,stepMm,rotR,counter)){
          messages.push("Direct extraction path found (no rotation needed).");
          return finish([start,goal],"direct");
        }
      }
    }
    messages.push("Direct extraction failed, running RRT-Connect...");
    // RRT-Connect for extraction (same as insertion, just reversed start/goal)
    const rng=mulberry32(1337);
    const lo=[-dims[0]/2,-dims[1]/2,-dims[2]/2], hi=[dims[0]/2,dims[1]/2,dims[2]/2];
    if(f.sign>0)hi[f.axis]=dims[f.axis]/2+box.wall+outDepth; else lo[f.axis]=-dims[f.axis]/2-box.wall-outDepth;
    const samplePose=()=>{
      const p=[lo[0]+rng()*(hi[0]-lo[0]),lo[1]+rng()*(hi[1]-lo[1]),lo[2]+rng()*(hi[2]-lo[2])];
      const r=rng();
      let q;
      if(r<0.3)q=AA_QUATS[(rng()*AA_QUATS.length)|0];
      else if(r<0.5)q=qMul(qAxisAngle([rng()*2-1,rng()*2-1,rng()*2-1],(rng()*0.8-0.4)),AA_QUATS[(rng()*AA_QUATS.length)|0]);
      else q=qRandom(rng);
      return{p,q};
    };
    const dist=(a,b)=>vLen(vSub(a.p,b.p))+qAngle(a.q,b.q)*rotR;
    const step=settings.rrtStep;
    const mkTree=(root)=>({nodes:[{pose:root,parent:-1}]});
    const nearest=(tree,pose)=>{let bi=0,bd=Infinity;for(let i=0;i<tree.nodes.length;i++){const d=dist(tree.nodes[i].pose,pose);if(d<bd){bd=d;bi=i;}}return bi;};
    const steer=(from,to)=>{const d=dist(from,to);if(d<=step)return to;const t=step/d;return{p:vLerp(from.p,to.p,t),q:qSlerp(from.q,to.q,t)};};
    const extend=(tree,target)=>{
      const ni=nearest(tree,target);
      const nn=tree.nodes[ni].pose;
      const np=steer(nn,target);
      if(!poseFree(np,half,panels,counter))return{res:"trapped"};
      if(!edgeFree(nn,np,half,panels,stepMm,rotR,counter))return{res:"trapped"};
      tree.nodes.push({pose:np,parent:ni});
      return{res:dist(np,target)<1e-6?"reached":"advanced",idx:tree.nodes.length-1};
    };
    const connect=(tree,target)=>{
      let r;
      do{r=extend(tree,target);}while(r.res==="advanced"&&!timeUp());
      return r;
    };
    const trace=(tree,idx)=>{const out=[];let i=idx;while(i!==-1){out.push(tree.nodes[i].pose);i=tree.nodes[i].parent;}return out;};
    let treeA=mkTree(startPoses[0]), treeB=mkTree(goals[0]);
    for(let gi=1;gi<Math.min(goals.length,4);gi++)treeB.nodes.push({pose:goals[gi],parent:-1});
    let iter=0;
    while(!timeUp()&&treeA.nodes.length+treeB.nodes.length<settings.maxNodes){
      iter++;
      if(iter%150===0){onProgress&&onProgress("RRT-Connect… nodes "+(treeA.nodes.length+treeB.nodes.length));await sleep(0);}
      const rand=samplePose();
      const rA=extend(treeA,rand);
      if(rA.res!=="trapped"){
        const newPose=treeA.nodes[rA.idx].pose;
        const rB=connect(treeB,newPose);
        if(rB.res==="reached"){
          const pathA=trace(treeA,rA.idx).reverse();
          const pathB=trace(treeB,rB.idx);
          let raw=pathA.concat(pathB);
          for(let s2=0;s2<120&&raw.length>2&&!timeUp();s2++){
            const i=1+((rng()*(raw.length-2))|0);
            const j=1+((rng()*(raw.length-2))|0);
            const a2=Math.min(i,j),b2=Math.max(i,j);
            if(b2-a2<2)continue;
            if(edgeFree(raw[a2],raw[b2],half,panels,stepMm,rotR,counter))
              raw=raw.slice(0,a2+1).concat(raw.slice(b2));
          }
          messages.push("Extraction path found via RRT-Connect.");
          return finish(raw,"rrt-connect");
        }
      }
      const tmp=treeA;treeA=treeB;treeB=tmp;
    }
    return {status:"inconclusive",path:null,messages,
      stats:{timeMs:performance.now()-t0,evaluated:counter.n,stage:"rrt-connect",nodes:treeA.nodes.length+treeB.nodes.length}};
  }

  // INSERTION MODE (original code)
  const goals=[];
  for(const q of gQuats){
    const pose={p:gPos,q};
    if(poseFree(pose,half,panels,counter))goals.push(pose);
  }
  if(goals.length===0){
    return {status:"no_goal",path:null,messages:["No collision-free final pose exists at the requested final position/orientation (including tolerance)."],
      stats:{timeMs:performance.now()-t0,evaluated:counter.n,stage:"goal check"}};
  }

  const finish=(rawPath,stage)=>{
    const path=densifyPath(rawPath,half,Math.min(stepMm,3));
    let tight={minAll:Infinity,minOpen:Infinity,minWall:Infinity,idx:0};
    path.forEach((pose,i)=>{
      const g=poseGapInfo(pose,half,panels,counter);
      if(g.minAll<tight.minAll)tight={...g,idx:i};
      tight.minOpen=Math.min(tight.minOpen,g.minOpen);
      tight.minWall=Math.min(tight.minWall,g.minWall);
    });
    return {status:"feasible",path,messages,
      stats:{timeMs:performance.now()-t0,evaluated:counter.n,stage,
        minClearance:tight.minAll,minOpening:tight.minOpen,minWall:tight.minWall,
        tightIdx:tight.idx,tightPose:path[tight.idx],finalPose:path[path.length-1]}};
  };

  /* ---- Stage 1: direct axis-aligned insertion ---- */
  onProgress&&onProgress("Stage 1: direct axis-aligned insertion…");
  for(const goal of goals){
    for(const q of AA_QUATS){
      if(timeUp())break;
      const s=startPose(cfg,q,outDepth);
      // waypoint just inside the opening at opening lateral position
      const inside={p:[0,0,0],q};
      inside.p[f.u]=usableOpening(opening).offU;
      inside.p[f.v]=usableOpening(opening).offV;
      const axes=qToAxes(q);
      const depthHalf=half[0]*Math.abs(axes[0][f.axis])+half[1]*Math.abs(axes[1][f.axis])+half[2]*Math.abs(axes[2][f.axis]);
      inside.p[f.axis]=f.sign*(dims[f.axis]/2-depthHalf-0.5);
      if(!poseFree(inside,half,panels,counter))continue;
      if(!edgeFree(s,inside,half,panels,stepMm,rotR,counter))continue;
      if(qAngle(q,goal.q)<1e-6){
        if(edgeFree(inside,goal,half,panels,stepMm,rotR,counter))
          {messages.push("Direct axis-aligned insertion succeeded (no rotation needed).");return finish([s,inside,goal],"direct");}
      }else{
        if(edgeFree(inside,goal,half,panels,stepMm,rotR,counter))
          {messages.push("Axis-aligned pass-through, then reorientation inside the box.");return finish([s,inside,goal],"direct+rotate");}
      }
    }
  }
  messages.push("Stage 1: no straight axis-aligned insertion found.");

  /* ---- Stage 2: systematic tilt sampling (screw-style insert) ---- */
  onProgress&&onProgress("Stage 2: systematic tilt sampling…");
  const tiltAngles=[];
  for(let a=5;a<=60;a+=5){tiltAngles.push(a*Math.PI/180,-a*Math.PI/180);}
  const uAxis=[0,0,0];uAxis[f.u]=1;
  const vAxis=[0,0,0];vAxis[f.v]=1;
  outer2:
  for(const goal of goals){
    for(const base of AA_QUATS){
      for(const axis of [uAxis,vAxis]){
        for(const ang of tiltAngles){
          if(timeUp())break outer2;
          const qTilt=qMul(qAxisAngle(axis,ang),base);
          const s=startPose(cfg,qTilt,outDepth);
          if(!poseFree(s,half,panels,counter))continue;
          // simultaneous translate+rotate from tilted-outside to goal
          if(edgeFree(s,goal,half,panels,stepMm,rotR,counter)){
            messages.push("Tilted insertion found: rotate "+ (ang*180/Math.PI).toFixed(0)+"° about the "+(axis===uAxis?"horizontal":"vertical")+" face axis while translating through the opening.");
            return finish([s,goal],"tilt-sweep");
          }
          // or: tilted pass to an intermediate pose near opening, then straighten
          const mid={p:[0,0,0],q:qTilt};
          mid.p[f.u]=usableOpening(opening).offU;mid.p[f.v]=usableOpening(opening).offV;
          mid.p[f.axis]=f.sign*(dims[f.axis]/2-rotR-0.5);
          if(mid.p[f.axis]*f.sign<-dims[f.axis]/2)continue;
          if(poseFree(mid,half,panels,counter)
            &&edgeFree(s,mid,half,panels,stepMm,rotR,counter)
            &&edgeFree(mid,goal,half,panels,stepMm,rotR,counter)){
            messages.push("Tilted pass-through with intermediate pose, then reorientation to final pose.");
            return finish([s,mid,goal],"tilt-sweep");
          }
        }
      }
    }
  }
  messages.push("Stage 2: systematic tilt sweep found no path.");

  /* ---- Stage 3: RRT-Connect ---- */
  onProgress&&onProgress("Stage 3: RRT-Connect randomized search…");
  const rng=mulberry32(1337);
  // sampling bounds: interior plus a corridor outside the opening face
  const lo=[-dims[0]/2,-dims[1]/2,-dims[2]/2], hi=[dims[0]/2,dims[1]/2,dims[2]/2];
  if(f.sign>0)hi[f.axis]=dims[f.axis]/2+box.wall+outDepth; else lo[f.axis]=-dims[f.axis]/2-box.wall-outDepth;
  const samplePose=()=>{
    const p=[lo[0]+rng()*(hi[0]-lo[0]),lo[1]+rng()*(hi[1]-lo[1]),lo[2]+rng()*(hi[2]-lo[2])];
    const r=rng();
    let q;
    if(r<0.3)q=AA_QUATS[(rng()*AA_QUATS.length)|0];
    else if(r<0.5)q=qMul(qAxisAngle([rng()*2-1,rng()*2-1,rng()*2-1],(rng()*0.8-0.4)),AA_QUATS[(rng()*AA_QUATS.length)|0]);
    else q=qRandom(rng);
    return{p,q};
  };
  const dist=(a,b)=>vLen(vSub(a.p,b.p))+qAngle(a.q,b.q)*rotR;
  const step=settings.rrtStep;
  const mkTree=(root)=>({nodes:[{pose:root,parent:-1}]});
  const nearest=(tree,pose)=>{let bi=0,bd=Infinity;for(let i=0;i<tree.nodes.length;i++){const d=dist(tree.nodes[i].pose,pose);if(d<bd){bd=d;bi=i;}}return bi;};
  const steer=(from,to)=>{const d=dist(from,to);if(d<=step)return to;const t=step/d;return{p:vLerp(from.p,to.p,t),q:qSlerp(from.q,to.q,t)};};
  const extend=(tree,target)=>{
    const ni=nearest(tree,target);
    const nn=tree.nodes[ni].pose;
    const np=steer(nn,target);
    if(!poseFree(np,half,panels,counter))return{res:"trapped"};
    if(!edgeFree(nn,np,half,panels,stepMm,rotR,counter))return{res:"trapped"};
    tree.nodes.push({pose:np,parent:ni});
    return{res:dist(np,target)<1e-6?"reached":"advanced",idx:tree.nodes.length-1};
  };
  const connect=(tree,target)=>{
    let r;
    do{r=extend(tree,target);}while(r.res==="advanced"&&!timeUp());
    return r;
  };
  const trace=(tree,idx)=>{const out=[];let i=idx;while(i!==-1){out.push(tree.nodes[i].pose);i=tree.nodes[i].parent;}return out;};

  const start=startPose(cfg,Q_ID,outDepth);
  const startFree=poseFree(start,half,panels,counter)?start:null;
  if(!startFree){messages.push("Warning: default start pose collides; using elevated start.");start.p[f.axis]+=f.sign*rotR;}
  let treeA=mkTree(start), treeB=mkTree(goals[0]);
  for(let gi=1;gi<Math.min(goals.length,4);gi++)treeB.nodes.push({pose:goals[gi],parent:-1});
  let iter=0;
  while(!timeUp()&&treeA.nodes.length+treeB.nodes.length<settings.maxNodes){
    iter++;
    if(iter%150===0){onProgress&&onProgress("Stage 3: RRT-Connect… nodes "+(treeA.nodes.length+treeB.nodes.length)+", checks "+counter.n);await sleep(0);}
    const rand=samplePose();
    const rA=extend(treeA,rand);
    if(rA.res!=="trapped"){
      const newPose=treeA.nodes[rA.idx].pose;
      const rB=connect(treeB,newPose);
      if(rB.res==="reached"){
        const pathA=trace(treeA,rA.idx).reverse();
        const pathB=trace(treeB,rB.idx);
        let raw=pathA.concat(pathB);
        // shortcut smoothing
        for(let s2=0;s2<120&&raw.length>2&&!timeUp();s2++){
          const i=1+((rng()*(raw.length-2))|0);
          const j=1+((rng()*(raw.length-2))|0);
          const a2=Math.min(i,j),b2=Math.max(i,j);
          if(b2-a2<2)continue;
          if(edgeFree(raw[a2],raw[b2],half,panels,stepMm,rotR,counter))
            raw=raw.slice(0,a2+1).concat(raw.slice(b2));
        }
        messages.push("RRT-Connect found a randomized collision-free path ("+iter+" iterations).");
        return finish(raw,"rrt-connect");
      }
    }
    const tmp=treeA;treeA=treeB;treeB=tmp;
  }
  messages.push("Stage 3: RRT-Connect exhausted the search budget ("+iter+" iterations, "+(treeA.nodes.length+treeB.nodes.length)+" nodes).");
  return {status:"inconclusive",path:null,messages,
    stats:{timeMs:performance.now()-t0,evaluated:counter.n,stage:"rrt-connect",nodes:treeA.nodes.length+treeB.nodes.length}};
}

/* ----------------------- clearance classification ----------------------- */
function classifyClearance(mm){
  if(mm<0.5)return{label:"too tight",tone:"bad",advice:"Below 0.5 mm effective clearance. Treat as not practically insertable without fixtures."};
  if(mm<1.5)return{label:"possible but difficult",tone:"warn",advice:"0.5–1.5 mm. Feasible with careful alignment; expect fiddly assembly."};
  if(mm<4)return{label:"reasonable",tone:"ok",advice:"1.5–4 mm. Normal hand-assembly clearance."};
  return{label:"comfortable",tone:"good",advice:"Over 4 mm of clearance along the whole path."};
}

/* ------------------------- lightweight unit tests ------------------------ */
function runUnitTests(){
  const results=[];
  const T=(name,fn)=>{try{const ok=fn();results.push({name,ok:!!ok});}catch(e){results.push({name,ok:false,err:String(e)});}};
  const base={
    box:{x:200,y:150,z:150,wall:5,face:"front"},
    opening:{w:120,h:100,offU:0,offV:0,radius:0,clearance:0.5},
    obj:{l:80,w:60,h:60,clearance:0.5,finalMode:"center",fx:0,fy:0,fz:0,anyOrientation:true},
  };
  T("direct axis-aligned fit detected",()=>prelimChecks(base).direct===true);
  T("object larger than interior rejected",()=>{
    const c=JSON.parse(JSON.stringify(base));c.obj.l=500;
    return prelimChecks(c).provablyImpossible.length>0;});
  T("opening outside face rejected",()=>{
    const c=JSON.parse(JSON.stringify(base));c.opening.offU=80;
    return validateConfig(c).length>0;});
  T("opening smaller than min dim => impossible",()=>{
    const c=JSON.parse(JSON.stringify(base));c.opening.w=40;c.opening.h=40;
    return prelimChecks(c).openingPossible===false;});
  T("wall panel collision detected",()=>{
    const panels=buildPanels(base.box,base.opening,"front");
    const pose={p:[0,0,base.box.z/2+base.box.wall/2],q:Q_ID}; // inside the front wall, off-opening object
    const pose2={p:[95,0,base.box.z/2],q:Q_ID};
    return !poseFree(pose2,[40,30,30],panels)&&poseFree({p:[0,0,0],q:Q_ID},[40,30,30],panels);});
  T("pose through opening centre is free",()=>{
    const panels=buildPanels(base.box,base.opening,"front");
    return poseFree({p:[0,0,base.box.z/2],q:Q_ID},[40,30,30],panels);});
  T("clearance reduces usable opening",()=>{
    const uo=usableOpening({...base.opening,clearance:5});
    return Math.abs(uo.w-110)<1e-9&&Math.abs(uo.h-90)<1e-9;});
  T("rotated (tilted) pose collision behaves",()=>{
    const panels=buildPanels(base.box,base.opening,"front");
    const q=qAxisAngle([1,0,0],Math.PI/4);
    // tall thin plate tilted in the opening plane should be free at centre
    return poseFree({p:[0,0,base.box.z/2],q},[50,5,5],panels);});
  return results;
}

/* ============================ SECTION 2 ============================ */
/* three.js viewport with custom orbit controls (no OrbitControls in r128 build) */

const C = {
  bg:"#10151b", panel:"#161d25", panel2:"#1b2430", line:"#28323f",
  ink:"#dbe4ee", dim:"#7d8b9c", faint:"#4c5a6a",
  amber:"#ffb454", object:"#ff7847", objectEdge:"#ffc9ae",
  glass:"#3d5a78", good:"#43d17e", bad:"#ff5f5f", warn:"#ffd166", accent:"#5ec8e5",
};

function makeTextSprite(text,color){
  const cv=document.createElement("canvas");
  const ctx=cv.getContext("2d");
  const fs=42;
  ctx.font=`600 ${fs}px ui-monospace, monospace`;
  const w=Math.ceil(ctx.measureText(text).width)+24;
  cv.width=w;cv.height=fs+20;
  const c2=cv.getContext("2d");
  c2.font=`600 ${fs}px ui-monospace, monospace`;
  c2.fillStyle=color;c2.textBaseline="middle";
  c2.fillText(text,12,cv.height/2);
  const tex=new THREE.CanvasTexture(cv);
  tex.minFilter=THREE.LinearFilter;
  const mat=new THREE.SpriteMaterial({map:tex,transparent:true,depthTest:false});
  const sp=new THREE.Sprite(mat);
  const scale=0.32;
  sp.scale.set(cv.width*scale,cv.height*scale,1);
  return sp;
}

function Viewport({cfg,result,tParam,mode,showLabels,wireframe,sectionCut,onResetRef}){
  const mountRef=useRef(null);
  const stateRef=useRef(null);

  // init once
  useEffect(()=>{
    const mount=mountRef.current;
    const renderer=new THREE.WebGLRenderer({antialias:true,alpha:false});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
    renderer.setClearColor(new THREE.Color(C.bg),1);
    mount.appendChild(renderer.domElement);
    const scene=new THREE.Scene();
    const camera=new THREE.PerspectiveCamera(42,1,1,20000);
    const amb=new THREE.AmbientLight(0xffffff,0.65);scene.add(amb);
    const key=new THREE.DirectionalLight(0xffffff,0.7);key.position.set(1,1.4,0.8);scene.add(key);
    const rim=new THREE.DirectionalLight(0x88aaff,0.25);rim.position.set(-1,-0.4,-1);scene.add(rim);
    const grid=new THREE.GridHelper(2000,40,new THREE.Color(C.line),new THREE.Color("#1a222c"));
    scene.add(grid);
    const enclosure=new THREE.Group();scene.add(enclosure);
    const dyn=new THREE.Group();scene.add(dyn);
    const st={renderer,scene,camera,enclosure,dyn,grid,
      sph:{theta:Math.PI/4,phi:Math.PI/3.1,r:600},drag:null,raf:0,target:new THREE.Vector3(0,0,0)};
    stateRef.current=st;

    const el=renderer.domElement;
    el.style.display="block";el.style.width="100%";el.style.height="100%";el.style.touchAction="none";
    const onDown=(e)=>{st.drag={x:e.clientX,y:e.clientY};el.setPointerCapture(e.pointerId);};
    const onMove=(e)=>{if(!st.drag)return;
      st.sph.theta-=(e.clientX-st.drag.x)*0.006;
      st.sph.phi=Math.max(0.08,Math.min(Math.PI-0.08,st.sph.phi-(e.clientY-st.drag.y)*0.006));
      st.drag={x:e.clientX,y:e.clientY};};
    const onUp=()=>{st.drag=null;};
    const onWheel=(e)=>{e.preventDefault();st.sph.r=Math.max(80,Math.min(8000,st.sph.r*(1+e.deltaY*0.001)));};
    el.addEventListener("pointerdown",onDown);
    el.addEventListener("pointermove",onMove);
    el.addEventListener("pointerup",onUp);
    el.addEventListener("wheel",onWheel,{passive:false});

    const resize=()=>{
      const w=mount.clientWidth,h=mount.clientHeight;
      renderer.setSize(w,h,false);
      camera.aspect=w/Math.max(1,h);camera.updateProjectionMatrix();};
    resize();
    const ro=new ResizeObserver(resize);ro.observe(mount);

    const loop=()=>{
      try{
        const {sph}=st;
        camera.position.set(
          st.target.x+sph.r*Math.sin(sph.phi)*Math.cos(sph.theta),
          st.target.y+sph.r*Math.cos(sph.phi),
          st.target.z+sph.r*Math.sin(sph.phi)*Math.sin(sph.theta));
        camera.lookAt(st.target);
        renderer.render(scene,camera);
      }catch(err){console.error("Viewport render error:",err);}
      st.raf=requestAnimationFrame(loop);};
    loop();
    if(onResetRef)onResetRef.current=()=>{st.sph={theta:Math.PI/4,phi:Math.PI/3.1,r:st.fitR||600};};
    return()=>{cancelAnimationFrame(st.raf);ro.disconnect();
      el.removeEventListener("pointerdown",onDown);el.removeEventListener("pointermove",onMove);
      el.removeEventListener("pointerup",onUp);el.removeEventListener("wheel",onWheel);
      renderer.dispose();mount.removeChild(el);};
  },[]);

  // rebuild static geometry when config / toggles change
  useEffect(()=>{
    const st=stateRef.current;if(!st)return;
    try{
    const {enclosure}=st;
    while(enclosure.children.length)enclosure.remove(enclosure.children[0]);
    const sane=(v)=>Number.isFinite(v)&&v>0?v:0.1;
    const box={...cfg.box,x:sane(cfg.box.x),y:sane(cfg.box.y),z:sane(cfg.box.z),wall:sane(cfg.box.wall)};
    const opening={...cfg.opening,w:sane(cfg.opening.w),h:sane(cfg.opening.h),
      offU:Number.isFinite(cfg.opening.offU)?cfg.opening.offU:0,offV:Number.isFinite(cfg.opening.offV)?cfg.opening.offV:0,
      radius:Math.max(0,Number.isFinite(cfg.opening.radius)?cfg.opening.radius:0),
      clearance:Math.max(0,Number.isFinite(cfg.opening.clearance)?cfg.opening.clearance:0)};
    const obj={...cfg.obj,l:sane(cfg.obj.l),w:sane(cfg.obj.w),h:sane(cfg.obj.h)};
    if(!FACES[box.face])box.face="front";
    const dims=[box.x,box.y,box.z];
    const f=FACES[box.face];
    const opposite={front:"back",back:"front",left:"right",right:"left",top:"bottom",bottom:"top"}[box.face];
    const panels=buildPanels(box,opening,box.face);
    const wallMat=new THREE.MeshStandardMaterial({
      color:new THREE.Color(C.glass),transparent:true,opacity:wireframe?0.08:0.28,
      roughness:0.4,metalness:0.1,side:THREE.DoubleSide,depthWrite:false});
    for(const p of panels){
      if(sectionCut&&p.name===opposite)continue;
      const g=new THREE.BoxGeometry(p.half[0]*2,p.half[1]*2,p.half[2]*2);
      const m=new THREE.Mesh(g,wallMat);
      m.position.set(...p.center);
      enclosure.add(m);
      const eg=new THREE.EdgesGeometry(g);
      const le=new THREE.LineSegments(eg,new THREE.LineBasicMaterial({color:new THREE.Color(wireframe?C.accent:C.line),transparent:true,opacity:wireframe?0.9:0.55}));
      le.position.copy(m.position);
      enclosure.add(le);
    }
    // interior wire box
    const ig=new THREE.BoxGeometry(box.x,box.y,box.z);
    const iw=new THREE.LineSegments(new THREE.EdgesGeometry(ig),new THREE.LineBasicMaterial({color:new THREE.Color(C.faint),transparent:true,opacity:0.5}));
    enclosure.add(iw);
    // opening outline (actual opening, before clearance) in amber
    const shape=[];
    const w2=opening.w/2,h2=opening.h/2;
    const corners=[[-w2,-h2],[w2,-h2],[w2,h2],[-w2,h2],[-w2,-h2]];
    const pts=corners.map(([a,b])=>{
      const p=[0,0,0];
      p[f.u]=opening.offU+a;p[f.v]=opening.offV+b;
      p[f.axis]=f.sign*(dims[f.axis]/2+box.wall+0.6);
      return new THREE.Vector3(...p);});
    const og=new THREE.BufferGeometry().setFromPoints(pts);
    enclosure.add(new THREE.Line(og,new THREE.LineBasicMaterial({color:new THREE.Color(C.amber),linewidth:2})));
    const pts2=pts.map(p=>{const q=p.clone();q.setComponent(f.axis,f.sign*(dims[f.axis]/2-0.6));return q;});
    enclosure.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts2),new THREE.LineBasicMaterial({color:new THREE.Color(C.amber),transparent:true,opacity:0.5})));
    // labels
    if(showLabels){
      const lx=makeTextSprite(`X ${box.x} mm`,C.dim);lx.position.set(0,-box.y/2-30,box.z/2+40);enclosure.add(lx);
      const ly=makeTextSprite(`Y ${box.y} mm`,C.dim);ly.position.set(box.x/2+50,0,box.z/2+30);enclosure.add(ly);
      const lz=makeTextSprite(`Z ${box.z} mm`,C.dim);lz.position.set(box.x/2+60,-box.y/2-20,0);enclosure.add(lz);
      const lo=makeTextSprite(`opening ${opening.w}×${opening.h}`,C.amber);
      const lop=[0,0,0];lop[f.u]=opening.offU;lop[f.v]=opening.offV+opening.h/2+22;lop[f.axis]=f.sign*(dims[f.axis]/2+box.wall+2);
      lo.position.set(...lop);enclosure.add(lo);
      const lob=makeTextSprite(`object ${obj.l}×${obj.w}×${obj.h}`,C.object);
      lob.position.set(0,box.y/2+55,0);enclosure.add(lob);
    }
    // grid + camera fit
    st.grid.position.y=-box.y/2-box.wall-1;
    const fit=Math.max(box.x,box.y,box.z)*2.6+Math.max(obj.l,obj.w,obj.h);
    st.fitR=fit;
    if(!st.userMoved)st.sph.r=fit;
    }catch(err){console.error("Viewport geometry rebuild error:",err);}
  },[JSON.stringify(cfg),showLabels,wireframe,sectionCut]);

  // dynamic: object pose, path, ghosts, tight marker
  useEffect(()=>{
    const st=stateRef.current;if(!st)return;
    try{
    const {dyn}=st;
    while(dyn.children.length)dyn.remove(dyn.children[0]);
    const sane=(v)=>Number.isFinite(v)&&v>0?v:0.1;
    const obj={...cfg.obj,l:sane(cfg.obj.l),w:sane(cfg.obj.w),h:sane(cfg.obj.h)};
    const geo=new THREE.BoxGeometry(obj.l,obj.w,obj.h);
    const mat=new THREE.MeshStandardMaterial({color:new THREE.Color(C.object),roughness:0.45,metalness:0.15});
    const mesh=new THREE.Mesh(geo,mat);
    const edges=new THREE.LineSegments(new THREE.EdgesGeometry(geo),new THREE.LineBasicMaterial({color:new THREE.Color(C.objectEdge)}));
    mesh.add(edges);
    dyn.add(mesh);

    const path=result&&result.path;
    let pose;
    if(path&&path.length&&path[0]&&path[0].p){
      const idxF=tParam*(path.length-1);
      const i0=Math.floor(idxF),i1=Math.min(path.length-1,i0+1),tt=idxF-i0;
      pose={p:vLerp(path[i0].p,path[i1].p,tt),q:qSlerp(path[i0].q,path[i1].q,tt)};
      // path line
      const lp=path.map(n=>new THREE.Vector3(...n.p));
      dyn.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(lp),
        new THREE.LineBasicMaterial({color:new THREE.Color(C.accent),transparent:true,opacity:0.8})));
      // ghosts: start + final
      const ghostMat=new THREE.MeshBasicMaterial({color:new THREE.Color(C.object),wireframe:true,transparent:true,opacity:0.25});
      for(const gp of [path[0],path[path.length-1]]){
        if(gp&&gp.p){
          const g=new THREE.Mesh(geo.clone(),ghostMat);
          g.position.set(...gp.p);g.quaternion.set(...gp.q);dyn.add(g);
        }
      }
      // tightest point marker
      if(result.stats&&result.stats.tightPose&&result.stats.tightPose.p){
        const tp=result.stats.tightPose;
        const s=new THREE.Mesh(new THREE.SphereGeometry(4,16,12),
          new THREE.MeshBasicMaterial({color:new THREE.Color(C.warn)}));
        s.position.set(...tp.p);dyn.add(s);
        const gt=new THREE.Mesh(geo.clone(),new THREE.MeshBasicMaterial({color:new THREE.Color(C.warn),wireframe:true,transparent:true,opacity:0.5}));
        gt.position.set(...tp.p);gt.quaternion.set(...tp.q);dyn.add(gt);
      }
    }else{
      // idle: show object outside the opening face (insertion) or inside center (extraction)
      const idleCfg=FACES[cfg.box.face]?cfg:{...cfg,box:{...cfg.box,face:"front"}};
      if(mode==="extraction"){
        pose={p:[0,0,0],q:Q_ID};  // center of box for extraction
      }else{
        pose=startPose(idleCfg,Q_ID,Math.max(obj.l,obj.w,obj.h)/2+(Number.isFinite(cfg.box.wall)?cfg.box.wall:5)+20);
      }
    }
    if(pose&&pose.p&&pose.p.every(Number.isFinite)){
      mesh.position.set(...pose.p);
      mesh.quaternion.set(pose.q[0],pose.q[1],pose.q[2],pose.q[3]);
    }
    }catch(err){console.error("Viewport pose update error:",err);}
  },[JSON.stringify(cfg),result,tParam,mode]);

  return <div ref={mountRef} style={{position:"absolute",inset:0}}/>;
}

/* ============================ SECTION 3 ============================ */
/* application UI */

const PRESETS=[
  {name:"1 · Easy direct fit",cfg:{
    box:{x:200,y:150,z:150,wall:5,face:"front"},
    opening:{w:120,h:100,offU:0,offV:0,radius:0,clearance:0.5},
    obj:{l:80,w:60,h:60,clearance:0.5,finalMode:"center",fx:0,fy:0,fz:0,anyOrientation:true}}},
  {name:"2 · Requires rotation",cfg:{
    box:{x:220,y:160,z:120,wall:5,face:"front"},
    opening:{w:150,h:100,offU:0,offV:0,radius:0,clearance:0.5},
    obj:{l:180,w:50,h:50,clearance:0.5,finalMode:"center",fx:0,fy:0,fz:0,anyOrientation:true}}},
  {name:"3 · Impossible: opening too small",cfg:{
    box:{x:200,y:150,z:150,wall:5,face:"front"},
    opening:{w:45,h:45,offU:0,offV:0,radius:0,clearance:0.5},
    obj:{l:80,w:60,h:60,clearance:0.5,finalMode:"center",fx:0,fy:0,fz:0,anyOrientation:true}}},
  {name:"4 · Passes opening, cannot fit inside",cfg:{
    box:{x:200,y:150,z:150,wall:5,face:"front"},
    opening:{w:120,h:100,offU:0,offV:0,radius:0,clearance:0.5},
    obj:{l:320,w:40,h:40,clearance:0.5,finalMode:"center",fx:0,fy:0,fz:0,anyOrientation:true}}},
  {name:"5 · Tight clearance",cfg:{
    box:{x:200,y:150,z:150,wall:5,face:"front"},
    opening:{w:85,h:65,offU:0,offV:0,radius:0,clearance:0.5},
    obj:{l:120,w:80,h:60,clearance:0.5,finalMode:"center",fx:0,fy:0,fz:0,anyOrientation:true}}},
];

const DEFAULT_SETTINGS={maxTimeMs:5000,stepMm:4,rrtStep:30,maxNodes:6000};

function eulerDeg(q){
  if(!q||q.length!==4)return[0,0,0];
  const e=new THREE.Euler().setFromQuaternion(new THREE.Quaternion(q[0],q[1],q[2],q[3]),"XYZ");
  return [e.x,e.y,e.z].map(a=>(a*180/Math.PI));
}
const fmt=(v,d=1)=>Number.isFinite(v)?v.toFixed(d):"—";

/* --- small UI atoms --- */
const S={
  label:{fontSize:10.5,letterSpacing:"0.09em",textTransform:"uppercase",color:C.dim,fontWeight:600},
  mono:{fontFamily:"ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"},
};
function Num({label,value,onChange,step=1,suffix="mm",min}){
  return(
    <label style={{display:"flex",alignItems:"center",justifyContent:"space-between",gap:8,padding:"3px 0"}}>
      <span style={{...S.label,flex:"1 1 auto"}}>{label}</span>
      <span style={{display:"flex",alignItems:"center",gap:5}}>
        <input type="number" value={value} step={step} min={min}
          onChange={e=>{const v=e.target.value===""?0:parseFloat(e.target.value);onChange(Number.isFinite(v)?v:0);}}
          style={{width:78,background:C.bg,border:`1px solid ${C.line}`,color:C.ink,
            borderRadius:4,padding:"5px 7px",fontSize:13,...S.mono,textAlign:"right",outline:"none"}}/>
        <span style={{fontSize:10,color:C.faint,width:20,...S.mono}}>{suffix}</span>
      </span>
    </label>);
}
function Section({title,children,defaultOpen=true}){
  const[open,setOpen]=useState(defaultOpen);
  return(
    <div style={{borderBottom:`1px solid ${C.line}`}}>
      <button onClick={()=>setOpen(o=>!o)} style={{width:"100%",display:"flex",justifyContent:"space-between",
        alignItems:"center",background:"none",border:"none",color:C.ink,cursor:"pointer",padding:"10px 2px",
        fontSize:11.5,letterSpacing:"0.12em",textTransform:"uppercase",fontWeight:700}}>
        {title}<span style={{color:C.faint}}>{open?"−":"+"}</span>
      </button>
      {open&&<div style={{padding:"2px 2px 12px"}}>{children}</div>}
    </div>);
}
function Chip({ok,unknown,children}){
  const col=unknown?C.warn:ok?C.good:C.bad;
  return <span style={{display:"inline-flex",alignItems:"center",gap:6,fontSize:12,color:C.ink}}>
    <span style={{width:8,height:8,borderRadius:99,background:col,boxShadow:`0 0 8px ${col}66`}}/>
    {children}</span>;
}

class ErrorBoundary extends React.Component{
  constructor(p){super(p);this.state={err:null};}
  static getDerivedStateFromError(err){return{err};}
  componentDidCatch(err,info){console.error("Cuboid Insertion Checker crashed:",err,info);}
  render(){
    if(this.state.err){
      const msg=this.state.err&&(this.state.err.message||String(this.state.err));
      const stack=this.state.err&&this.state.err.stack?String(this.state.err.stack).split("\n").slice(0,4).join("\n"):"";
      return(
        <div style={{position:"fixed",inset:0,display:"flex",alignItems:"center",justifyContent:"center",
          background:"#10151b",color:"#dbe4ee",fontFamily:"system-ui, sans-serif",padding:24}}>
          <div style={{maxWidth:560,textAlign:"center"}}>
            <div style={{fontSize:15,fontWeight:800,marginBottom:8,color:"#ff5f5f"}}>Something went wrong rendering the result</div>
            <div style={{fontSize:12.5,color:"#7d8b9c",lineHeight:1.5,marginBottom:10}}>
              The inputs were kept — press Try again to re-render. If it repeats, the details below identify the cause.
            </div>
            <pre style={{textAlign:"left",fontSize:10.5,color:"#ffd166",background:"#161d25",border:"1px solid #28323f",
              borderRadius:6,padding:"10px 12px",overflowX:"auto",whiteSpace:"pre-wrap",marginBottom:16}}>{msg+"\n"+stack}</pre>
            <button onClick={()=>this.setState({err:null})}
              style={{background:"#5ec8e522",border:"1px solid #5ec8e5",color:"#5ec8e5",borderRadius:6,
                padding:"8px 16px",fontSize:12.5,cursor:"pointer",fontWeight:700}}>
              Try again
            </button>
          </div>
        </div>);
    }
    return this.props.children;
  }
}

function App(){
  const[cfg,setCfg]=useState(PRESETS[0].cfg);
  const[settings,setSettings]=useState(DEFAULT_SETTINGS);
  const[result,setResult]=useState(null);
  const[running,setRunning]=useState(false);
  const[progress,setProgress]=useState("");
  const[tParam,setT]=useState(0);
  const[playing,setPlaying]=useState(false);
  const[showLabels,setShowLabels]=useState(true);
  const[wireframe,setWireframe]=useState(false);
  const[sectionCut,setSectionCut]=useState(false);
  const[showTests,setShowTests]=useState(false);
  const[copied,setCopied]=useState(false);
  const[leftSidebarOpen,setLeftSidebarOpen]=useState(true);
  const[rightSidebarOpen,setRightSidebarOpen]=useState(true);
  const[mode,setMode]=useState("insertion");  // "insertion" or "extraction"
  const[customStartPos,setCustomStartPos]=useState(false);
  const[startX,setStartX]=useState(0);
  const[startY,setStartY]=useState(0);
  const[startZ,setStartZ]=useState(0);
  const[startRot,setStartRot]=useState("identity");  // "identity", "random", or custom quat
  const resetCamRef=useRef(null);
  const fileRef=useRef(null);

  const errors=useMemo(()=>validateConfig(cfg),[cfg]);
  const prelim=useMemo(()=>errors.length?null:prelimChecks(cfg),[cfg,errors]);
  const tests=useMemo(()=>runUnitTests(),[]);

  const upd=(group,key,val)=>{setResult(null);setCfg(c=>({...c,[group]:{...c[group],[key]:val}}));};

  // playback
  useEffect(()=>{
    if(!playing||!result||!result.path)return;
    let raf;const start=performance.now();const t0=tParam>=0.999?0:tParam;
    const dur=Math.max(2500,result.path.length*12);
    const tick=(now)=>{
      const t=Math.min(1,t0+(now-start)/dur);
      setT(t);
      if(t<1)raf=requestAnimationFrame(tick);else setPlaying(false);};
    raf=requestAnimationFrame(tick);
    return()=>cancelAnimationFrame(raf);
  },[playing,result]);

  const run=async()=>{
    if(errors.length)return;
    setRunning(true);setResult(null);setT(0);setPlaying(false);
    setProgress("Preparing solver…");
    await new Promise(r=>setTimeout(r,30));
    if(prelim&&prelim.provablyImpossible.length){
      setResult({status:"impossible",path:null,messages:prelim.provablyImpossible,
        stats:{timeMs:0,evaluated:0,stage:"analytic"}});
      setRunning(false);setProgress("");return;
    }
    try{
      const res=await solveInsertion(cfg,settings,setProgress,mode==="extraction");
      setResult(res);
      if(res.status==="feasible"&&res.path&&res.path.length){setT(0);setPlaying(true);}
    }catch(err){
      console.error("Solver error:",err);
      setResult({status:"error",path:null,messages:["Solver error: "+(err&&err.message?err.message:String(err)),
        "Try adjusting the inputs slightly or reducing the search settings."],stats:{timeMs:0,evaluated:0,stage:"error"}});
    }
    setRunning(false);setProgress("");
  };

  const statusInfo=(()=>{
    if(errors.length)return{tone:C.bad,head:"INVALID INPUT",sub:"Fix the highlighted input errors."};
    if(running)return{tone:C.accent,head:"SOLVING…",sub:progress||"Searching for a collision-free path."};
    if(!result)return{tone:C.faint,head:"READY",sub:"Preliminary checks below · run the full solver to test an insertion path."};
    if(result.status==="feasible")return{tone:C.good,head:"INSERTION FEASIBLE",sub:"A continuous collision-free path was found under the selected tolerances."};
    if(result.status==="error")return{tone:C.bad,head:"SOLVER ERROR",sub:"The solver hit an unexpected condition — see the log below."};
    if(result.status==="impossible"||result.status==="no_goal")return{tone:C.bad,head:"NO PATH — PROVEN INFEASIBLE",sub:"Analytic checks rule out any insertion path with these dimensions and tolerances."};
    return{tone:C.warn,head:"INCONCLUSIVE — SEARCH LIMIT REACHED",sub:"No valid path found within the search limit; this is not a mathematical proof of impossibility."};
  })();

  const summaryText=()=>{
    const L=[];
    L.push("CUBOID INSERTION CHECKER — result summary");
    L.push(`Box interior: ${cfg.box.x} × ${cfg.box.y} × ${cfg.box.z} mm, wall ${cfg.box.wall} mm, opening on ${cfg.box.face}`);
    L.push(`Opening: ${cfg.opening.w} × ${cfg.opening.h} mm @ (${cfg.opening.offU}, ${cfg.opening.offV}), r=${cfg.opening.radius}, clearance ${cfg.opening.clearance} mm`);
    L.push(`Object: ${cfg.obj.l} × ${cfg.obj.w} × ${cfg.obj.h} mm, tolerance ${cfg.obj.clearance} mm`);
    L.push(`Status: ${statusInfo.head}`);
    if(result&&result.stats){
      const s=result.stats;
      if(result.status==="feasible"&&s.tightPose&&s.finalPose){
        L.push(`Min opening clearance: ${fmt(s.minOpening,2)} mm · min wall clearance: ${fmt(s.minWall,2)} mm`);
        L.push(`Clearance class: ${classifyClearance(s.minClearance).label}`);
        const e=eulerDeg(s.tightPose.q);
        L.push(`Orientation at tightest point (XYZ): ${e.map(a=>fmt(a,1)+"°").join(" / ")}`);
        const ef=eulerDeg(s.finalPose.q);
        L.push(`Final position: (${s.finalPose.p.map(v=>fmt(v,1)).join(", ")}) mm · final orientation: ${ef.map(a=>fmt(a,1)+"°").join(" / ")}`);
        L.push(`Solver stage: ${s.stage}`);
      }
      L.push(`Computation: ${fmt(s.timeMs,0)} ms · ${(s.evaluated||0).toLocaleString()} configurations evaluated`);
    }
    if(result&&result.messages)result.messages.forEach(m=>L.push("- "+m));
    return L.join("\n");
  };
  const copySummary=async()=>{
    const txt=summaryText();
    try{await navigator.clipboard.writeText(txt);}
    catch{const ta=document.createElement("textarea");ta.value=txt;document.body.appendChild(ta);ta.select();document.execCommand("copy");ta.remove();}
    setCopied(true);setTimeout(()=>setCopied(false),1500);
  };
  const exportJSON=()=>{
    const blob=new Blob([JSON.stringify({version:1,cfg,settings},null,2)],{type:"application/json"});
    const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="insertion-check.json";a.click();
    URL.revokeObjectURL(a.href);};
  const importJSON=(e)=>{
    const file=e.target.files&&e.target.files[0];if(!file)return;
    const rd=new FileReader();
    rd.onload=()=>{try{
      const d=JSON.parse(rd.result);
      if(d.cfg){
        const base=JSON.parse(JSON.stringify(PRESETS[0].cfg));
        setCfg({box:{...base.box,...(d.cfg.box||{})},opening:{...base.opening,...(d.cfg.opening||{})},obj:{...base.obj,...(d.cfg.obj||{})}});
      }
      if(d.settings)setSettings(s=>({...s,...d.settings}));
      setResult(null);setT(0);
    }catch{alert("Could not read that file as a project JSON.");}};
    rd.readAsText(file);e.target.value="";};

  const clsInfo=result&&result.status==="feasible"?classifyClearance(result.stats.minClearance):null;

  return(
  <div style={{position:"fixed",inset:0,display:"flex",flexDirection:"column",background:C.bg,color:C.ink,
    fontFamily:"'Segoe UI', system-ui, -apple-system, sans-serif",fontSize:13,overflow:"hidden"}}>
    {/* header */}
    <header style={{display:"flex",alignItems:"center",gap:14,padding:"10px 16px",
      borderBottom:`1px solid ${C.line}`,background:C.panel,flexWrap:"wrap"}}>
      <img src="/dessina-logo.png" alt="Dessina" style={{height:32,marginRight:8}}/>
      <div style={{display:"flex",alignItems:"baseline",gap:10}}>
        <span style={{fontWeight:800,letterSpacing:"0.04em",fontSize:15}}>CUBOID INSERTION CHECKER</span>
        <span style={{...S.mono,fontSize:10,color:C.faint,border:`1px solid ${C.line}`,padding:"2px 6px",borderRadius:3}}>mm · 6-DOF path solver</span>
      </div>
      <div style={{flex:1}}/>
      <div style={{display:"flex",alignItems:"center",gap:8,padding:"6px 12px",border:`1px solid ${C.line}`,borderRadius:6,background:C.panel2}}>
        <span style={{fontSize:11,fontWeight:600,color:mode==="insertion"?C.accent:C.dim,letterSpacing:"0.06em"}}>INSERT</span>
        <button onClick={()=>setMode(mode==="insertion"?"extraction":"insertion")}
          style={{width:50,height:26,borderRadius:13,border:"none",background:mode==="insertion"?C.good:C.bad,
            cursor:"pointer",position:"relative",transition:"all 0.3s",display:"flex",alignItems:"center",
            padding:"2px 2px"}}>
          <div style={{width:22,height:22,borderRadius:11,background:"white",transition:"transform 0.3s",
            transform:mode==="extraction"?"translateX(24px)":"translateX(0)",boxShadow:"0 2px 4px rgba(0,0,0,0.2)"}}/>
        </button>
        <span style={{fontSize:11,fontWeight:600,color:mode==="extraction"?C.accent:C.dim,letterSpacing:"0.06em"}}>EXTRACT</span>
      </div>
      <button onClick={run} disabled={running||errors.length>0}
        style={{padding:"8px 16px",borderRadius:6,border:"none",cursor:running?"wait":"pointer",
          background:running?C.line:C.accent,color:"#08131a",fontWeight:800,letterSpacing:"0.06em",fontSize:12}}>
        {running?"SOLVING…":"RUN"}
      </button>
      <select value={-1} onChange={e=>{const i=+e.target.value;if(i>=0){setCfg(JSON.parse(JSON.stringify(PRESETS[i].cfg)));setResult(null);setT(0);}}}
        style={{background:C.bg,color:C.ink,border:`1px solid ${C.line}`,borderRadius:4,padding:"6px 8px",fontSize:12}}>
        <option value={-1}>Load example preset…</option>
        {PRESETS.map((p,i)=><option key={i} value={i}>{p.name}</option>)}
      </select>
      <button onClick={exportJSON} style={btn()}>Export JSON</button>
      <button onClick={()=>fileRef.current&&fileRef.current.click()} style={btn()}>Import JSON</button>
      <input ref={fileRef} type="file" accept=".json,application/json" style={{display:"none"}} onChange={importJSON}/>
      <button onClick={copySummary} style={btn(true)}>{copied?"Copied ✓":"Copy result summary"}</button>
    </header>

    {/* main grid */}
    <div style={{flex:1,display:"grid",gridTemplateColumns:leftSidebarOpen&&rightSidebarOpen?"288px 1fr 316px":leftSidebarOpen?"288px 1fr":rightSidebarOpen?"1fr 316px":"1fr",minHeight:0}}>
      {/* left: inputs */}
      {leftSidebarOpen&&<aside style={{overflowY:"auto",borderRight:`1px solid ${C.line}`,background:C.panel,padding:"4px 14px 20px"}}>
        <Section title="Enclosure (internal)">
          <Num label="Width X" value={cfg.box.x} onChange={v=>upd("box","x",v)}/>
          <Num label="Height Y" value={cfg.box.y} onChange={v=>upd("box","y",v)}/>
          <Num label="Depth Z" value={cfg.box.z} onChange={v=>upd("box","z",v)}/>
          <Num label="Wall thickness" value={cfg.box.wall} onChange={v=>upd("box","wall",v)}/>
          <label style={{display:"flex",justifyContent:"space-between",alignItems:"center",padding:"4px 0"}}>
            <span style={S.label}>Opening face</span>
            <select value={cfg.box.face} onChange={e=>upd("box","face",e.target.value)}
              style={{background:C.bg,color:C.ink,border:`1px solid ${C.line}`,borderRadius:4,padding:"5px 6px",fontSize:12}}>
              {Object.entries(FACES).map(([k,f])=><option key={k} value={k}>{f.label}</option>)}
            </select>
          </label>
        </Section>
        <Section title="Opening">
          <Num label="Width" value={cfg.opening.w} onChange={v=>upd("opening","w",v)}/>
          <Num label="Height" value={cfg.opening.h} onChange={v=>upd("opening","h",v)}/>
          <Num label="Horiz. offset" value={cfg.opening.offU} onChange={v=>upd("opening","offU",v)}/>
          <Num label="Vert. offset" value={cfg.opening.offV} onChange={v=>upd("opening","offV",v)}/>
          <Num label="Corner radius" value={cfg.opening.radius} onChange={v=>upd("opening","radius",v)}/>
          <Num label="Clearance margin" value={cfg.opening.clearance} step={0.1} onChange={v=>upd("opening","clearance",v)}/>
        </Section>
        <Section title="Inserted object">
          <Num label="Length" value={cfg.obj.l} onChange={v=>upd("obj","l",v)}/>
          <Num label="Width" value={cfg.obj.w} onChange={v=>upd("obj","w",v)}/>
          <Num label="Height" value={cfg.obj.h} onChange={v=>upd("obj","h",v)}/>
          <Num label="Tolerance" value={cfg.obj.clearance} step={0.1} onChange={v=>upd("obj","clearance",v)}/>
          <label style={{display:"flex",justifyContent:"space-between",alignItems:"center",padding:"4px 0"}}>
            <span style={S.label}>Final position</span>
            <select value={cfg.obj.finalMode} onChange={e=>upd("obj","finalMode",e.target.value)}
              style={{background:C.bg,color:C.ink,border:`1px solid ${C.line}`,borderRadius:4,padding:"5px 6px",fontSize:12}}>
              <option value="center">Centred in box</option>
              <option value="manual">Manual X/Y/Z</option>
            </select>
          </label>
          {cfg.obj.finalMode==="manual"&&<>
            <Num label="Final X" value={cfg.obj.fx} onChange={v=>upd("obj","fx",v)}/>
            <Num label="Final Y" value={cfg.obj.fy} onChange={v=>upd("obj","fy",v)}/>
            <Num label="Final Z" value={cfg.obj.fz} onChange={v=>upd("obj","fz",v)}/>
          </>}
          <label style={{display:"flex",justifyContent:"space-between",alignItems:"center",padding:"6px 0"}}>
            <span style={S.label}>Final orientation</span>
            <select value={cfg.obj.anyOrientation?"any":"aa"} onChange={e=>upd("obj","anyOrientation",e.target.value==="any")}
              style={{background:C.bg,color:C.ink,border:`1px solid ${C.line}`,borderRadius:4,padding:"5px 6px",fontSize:12}}>
              <option value="aa">Axis-aligned</option>
              <option value="any">Any orientation</option>
            </select>
          </label>
        </Section>
        <Section title="Solver settings" defaultOpen={false}>
          <Num label="Max search time" value={settings.maxTimeMs/1000} step={1} suffix="s" min={1}
            onChange={v=>setSettings(s=>({...s,maxTimeMs:Math.max(1,v)*1000}))}/>
          <Num label="Collision step" value={settings.stepMm} step={0.5} min={0.5}
            onChange={v=>setSettings(s=>({...s,stepMm:Math.max(0.5,v)}))}/>
          <Num label="RRT step size" value={settings.rrtStep} step={5} min={5}
            onChange={v=>setSettings(s=>({...s,rrtStep:Math.max(5,v)}))}/>
          <Num label="Max RRT nodes" value={settings.maxNodes} step={500} suffix="" min={500}
            onChange={v=>setSettings(s=>({...s,maxNodes:Math.max(500,v)}))}/>
        </Section>
        {errors.length>0&&<div style={{marginTop:12,padding:10,border:`1px solid ${C.bad}55`,borderRadius:6,background:`${C.bad}12`}}>
          {errors.map((e,i)=><div key={i} style={{color:C.bad,fontSize:12,padding:"2px 0"}}>• {e}</div>)}
        </div>}
        <button onClick={run} disabled={running||errors.length>0}
          style={{marginTop:14,width:"100%",padding:"11px 0",borderRadius:6,border:"none",cursor:running?"wait":"pointer",
            background:running?C.line:C.accent,color:"#08131a",fontWeight:800,letterSpacing:"0.06em",fontSize:13}}>
          {running?"SOLVING…":"RUN FULL 3D SOLVER"}
        </button>
        <button onClick={()=>setShowTests(t=>!t)} style={{...btn(),width:"100%",marginTop:8}}>
          {showTests?"Hide":"Show"} unit tests ({tests.filter(t=>t.ok).length}/{tests.length} passing)
        </button>
        {showTests&&<div style={{marginTop:8,fontSize:11.5,...S.mono}}>
          {tests.map((t,i)=><div key={i} style={{color:t.ok?C.good:C.bad,padding:"2px 0"}}>{t.ok?"✓":"✗"} {t.name}</div>)}
        </div>}
      </aside>}

      {/* centre: viewport */}
      <main style={{position:"relative",minWidth:0,background:C.bg}}>
        <Viewport cfg={cfg} result={result} tParam={tParam} mode={mode}
          showLabels={showLabels} wireframe={wireframe} sectionCut={sectionCut} onResetRef={resetCamRef}/>
        {/* status lamp */}
        <div style={{position:"absolute",top:12,left:12,right:12,display:"flex",gap:10,alignItems:"center",
          background:`${C.panel}e6`,border:`1px solid ${C.line}`,borderLeft:`4px solid ${statusInfo.tone}`,
          borderRadius:6,padding:"9px 14px",backdropFilter:"blur(4px)"}}>
          <span style={{width:11,height:11,borderRadius:99,background:statusInfo.tone,boxShadow:`0 0 12px ${statusInfo.tone}`}}/>
          <div>
            <div style={{fontWeight:800,letterSpacing:"0.08em",fontSize:12.5}}>{statusInfo.head}</div>
            <div style={{color:C.dim,fontSize:11.5}}>{statusInfo.sub}</div>
          </div>
        </div>
        {/* view toggles + sidebar toggles */}
        <div style={{position:"absolute",top:70,right:12,display:"flex",flexDirection:"column",gap:6}}>
          {[["Labels",showLabels,setShowLabels],["Wireframe",wireframe,setWireframe],["Section cut",sectionCut,setSectionCut]].map(([n,v,set])=>(
            <button key={n} onClick={()=>set(x=>!x)} style={{...btn(v),fontSize:11}}>{n}</button>))}
          <button onClick={()=>resetCamRef.current&&resetCamRef.current()} style={{...btn(),fontSize:11}}>Reset camera</button>
          <div style={{height:1,background:C.line,margin:"4px 0"}}/>
          <button onClick={()=>setLeftSidebarOpen(x=>!x)} style={{...btn(!leftSidebarOpen),fontSize:11}}>
            {leftSidebarOpen?"Hide inputs":"Show inputs"}
          </button>
          <button onClick={()=>setRightSidebarOpen(x=>!x)} style={{...btn(!rightSidebarOpen),fontSize:11}}>
            {rightSidebarOpen?"Hide output":"Show output"}
          </button>
        </div>
        {/* timeline */}
        {result&&result.path&&<div style={{position:"absolute",left:12,right:12,bottom:12,display:"flex",gap:10,alignItems:"center",
          background:`${C.panel}e6`,border:`1px solid ${C.line}`,borderRadius:6,padding:"8px 12px"}}>
          <button onClick={()=>setPlaying(p=>!p)} style={{...btn(true),minWidth:64}}>{playing?"Pause":"Play"}</button>
          <input type="range" min={0} max={1} step={0.0005} value={tParam}
            onChange={e=>{setPlaying(false);setT(parseFloat(e.target.value));}}
            style={{flex:1,accentColor:C.accent}}/>
          <span style={{...S.mono,fontSize:11,color:C.dim,minWidth:110,textAlign:"right"}}>
            step {Math.round(tParam*(result.path.length-1))+1} / {result.path.length}</span>
        </div>}
      </main>

      {/* right: results */}
      {rightSidebarOpen&&<aside style={{overflowY:"auto",borderLeft:`1px solid ${C.line}`,background:C.panel,padding:"12px 14px 20px"}}>
        <div style={{...S.label,marginBottom:8}}>Result</div>
        <div style={{border:`1px solid ${statusInfo.tone}55`,background:`${statusInfo.tone}10`,borderRadius:8,padding:"12px 12px"}}>
          <div style={{fontWeight:800,color:statusInfo.tone,letterSpacing:"0.06em",fontSize:13}}>{statusInfo.head}</div>
          <div style={{color:C.dim,fontSize:12,marginTop:4,lineHeight:1.45}}>{statusInfo.sub}</div>
        </div>
        {result&&result.status==="feasible"&&result.stats&&result.stats.tightPose&&result.stats.finalPose&&(()=>{const s=result.stats;const eT=eulerDeg(s.tightPose.q);const eF=eulerDeg(s.finalPose.q);return(
          <div style={{marginTop:12,display:"flex",flexDirection:"column",gap:10}}>
            <Stat k="Min opening clearance" v={`${fmt(s.minOpening,2)} mm`}/>
            <Stat k="Min wall clearance" v={`${fmt(s.minWall,2)} mm`}/>
            <div style={{border:`1px solid ${C.line}`,borderRadius:6,padding:"9px 10px"}}>
              <div style={S.label}>Practical clearance class</div>
              <div style={{color:clsInfo.tone==="bad"?C.bad:clsInfo.tone==="warn"?C.warn:clsInfo.tone==="ok"?C.accent:C.good,
                fontWeight:800,fontSize:14,margin:"4px 0 2px",textTransform:"uppercase",letterSpacing:"0.05em"}}>{clsInfo.label}</div>
              <div style={{color:C.dim,fontSize:11.5,lineHeight:1.45}}>{clsInfo.advice}</div>
            </div>
            <Stat k="Orientation @ tightest point" v={eT.map(a=>fmt(a,1)+"°").join(" · ")} mono/>
            <Stat k="Final position (X,Y,Z)" v={s.finalPose.p.map(v2=>fmt(v2,1)).join(" · ")+" mm"} mono/>
            <Stat k="Final orientation (XYZ)" v={eF.map(a=>fmt(a,1)+"°").join(" · ")} mono/>
            <Stat k="Solver stage" v={s.stage} mono/>
            <Stat k="Computation time" v={`${fmt(s.timeMs,0)} ms`} mono/>
            <Stat k="Configurations evaluated" v={s.evaluated.toLocaleString()} mono/>
            <button onClick={()=>{if(result.path&&s.tightIdx!==undefined)setT(s.tightIdx/(result.path.length-1));}} style={btn()}>Jump to tightest point</button>
          </div>);})()}
        {result&&result.status!=="feasible"&&result.stats&&(
          <div style={{marginTop:12,display:"flex",flexDirection:"column",gap:10}}>
            <Stat k="Computation time" v={`${fmt(result.stats.timeMs,0)} ms`} mono/>
            <Stat k="Configurations evaluated" v={(result.stats.evaluated||0).toLocaleString()} mono/>
            {result.stats.nodes&&<Stat k="RRT nodes" v={result.stats.nodes.toLocaleString()} mono/>}
          </div>)}
        {result&&result.messages.length>0&&<div style={{marginTop:14}}>
          <div style={{...S.label,marginBottom:6}}>Solver log</div>
          {result.messages.map((m,i)=><div key={i} style={{fontSize:11.5,color:C.dim,padding:"3px 0",lineHeight:1.4,borderBottom:`1px dashed ${C.line}`}}>{m}</div>)}
        </div>}
        <div style={{marginTop:16,padding:"10px 11px",border:`1px solid ${C.line}`,borderRadius:6,background:C.panel2,
          fontSize:11,color:C.dim,lineHeight:1.5}}>
          <b style={{color:C.ink}}>Accuracy note.</b> This calculator uses numerical geometric path planning. A successful
          result demonstrates a collision-free path under the selected tolerances. A failed search may mean the
          insertion is impossible or that the solver needs more search time or finer sampling. Validate critical
          production cases using CAD interference checks or physical testing.
        </div>
      </aside>}
    </div>

    {/* bottom: step-by-step checks */}
    <footer style={{borderTop:`1px solid ${C.line}`,background:C.panel,padding:"10px 16px",
      display:"flex",gap:26,alignItems:"flex-start",flexWrap:"wrap"}}>
      {prelim?(<>
        <div style={{minWidth:210}}>
          <div style={{...S.label,marginBottom:6}}>1 · Direct axis-aligned insertion</div>
          <Chip ok={prelim.direct}>{prelim.direct?"Straight pass-through possible":"No straight pass-through"}</Chip>
        </div>
        <div style={{minWidth:200}}>
          <div style={{...S.label,marginBottom:6}}>2 · Opening feasibility</div>
          <Chip ok={prelim.crossFits} unknown={!prelim.crossFits&&prelim.openingPossible}>
            {prelim.crossFits?"A cross-section fits the usable opening":prelim.openingPossible?"Only a tilted section might fit":"No cross-section can fit"}</Chip>
          <div style={{...S.mono,fontSize:10.5,color:C.faint,marginTop:4}}>usable {fmt(prelim.usable.openW)} × {fmt(prelim.usable.openH)} mm</div>
        </div>
        <div style={{minWidth:190}}>
          <div style={{...S.label,marginBottom:6}}>3 · Final containment</div>
          <Chip ok={prelim.containAA} unknown={!prelim.containAA&&prelim.containNecessary}>
            {prelim.containAA?"Fits interior axis-aligned":prelim.containNecessary?"Might fit only rotated":"Cannot fit inside"}</Chip>
        </div>
        <div style={{flex:"1 1 280px",fontSize:11,color:C.dim,lineHeight:1.5,maxWidth:520}}>
          <b style={{color:C.ink}}>4 · Note:</b> passing these preliminary checks does <b>not</b> guarantee a valid insertion
          path — the object must also be manoeuvred through the opening without collision. Run the full 3-D solver
          (direct → tilt sweep → RRT-Connect) to test an actual path.
        </div>
      </>):(<div style={{fontSize:12,color:C.bad}}>Preliminary checks unavailable — fix input errors first.</div>)}
    </footer>
  </div>);
}

function Stat({k,v,mono}){
  return(<div style={{display:"flex",justifyContent:"space-between",gap:10,borderBottom:`1px dashed ${C.line}`,padding:"4px 0"}}>
    <span style={{...S.label}}>{k}</span>
    <span style={{fontSize:12.5,color:C.ink,...(mono?S.mono:{}),textAlign:"right"}}>{v}</span>
  </div>);
}
function btn(primary){
  return{background:primary?`${C.accent}22`:C.panel2,border:`1px solid ${primary?C.accent:C.line}`,
    color:primary?C.accent:C.ink,borderRadius:5,padding:"6px 11px",fontSize:12,cursor:"pointer",fontWeight:600};
}

export default function AppWithBoundary(){
  return <ErrorBoundary><App/></ErrorBoundary>;
}
