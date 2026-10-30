import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import vm from 'node:vm';

const root=fileURLToPath(new URL('..',import.meta.url));
const read=name=>fs.readFileSync(path.join(root,'site',name),'utf8');
async function boot(prefix,search='') {
  const page='https://official.example'+prefix+'/';
  const elements=new Map();
  function get(id) {
    if(!elements.has(id)) {
      const events=new Map();
      elements.set(id,{innerHTML:'',textContent:'',value:'',hidden:false,disabled:false,
        querySelectorAll(){return [];},addEventListener(n,f){events.set(n,f);},
        dispatch(n,e){events.get(n)?.(e);},showModal(){this.open=true;},close(){this.open=false;}});
    }
    return elements.get(id);
  }
  const requests=[];
  const location={href:page+search,origin:'https://official.example',search};
  const env={URL,URLSearchParams,console,location,
    document:{baseURI:page,currentScript:{src:page+'app-path.js'},getElementById:get},
    history:{replaceState(a,b,u){location.href=String(u);}},window:{addEventListener(){}},
    fetch:async value=>{
      const u=new URL(value,page);requests.push(u.pathname);
      assert.ok(u.pathname.startsWith(prefix+'/'),u.pathname);
      const relative=u.pathname.slice(prefix.length+1);
      const file=path.join(root,'site',relative);
      return {ok:fs.existsSync(file),json:async()=>JSON.parse(fs.readFileSync(file,'utf8'))};
    }};
  vm.runInNewContext(read('app-path.js')+'\n'+read('ranking-model.js')+'\n'+read('ranking.js'),env);
  await new Promise(r=>setImmediate(r));
  return {get,env,requests};
}

test('actual rankings and all data requests work at root and arbitrary subpaths',async()=>{
  for(const prefix of ['', '/leaderboard','/research/leaderboard']) {
    const app=await boot(prefix);
    assert.match(app.get('leaderboard-wrap').innerHTML,/80\.15/);
    assert.match(app.get('leaderboard-wrap').innerHTML,/Attend &amp; Discriminate/);
    assert.equal(app.requests.length,6);
    assert.equal(app.env.IoTBenchPath('/downloads/a.zip'),'https://official.example'+prefix+'/downloads/a.zip');
    assert.equal(app.env.IoTBenchPath('data/a.json'),'https://official.example'+prefix+'/data/a.json');
    assert.equal(app.env.IoTBenchPath('https://paper.example/paper'),'https://paper.example/paper');
  }
});
test('result links retain the deployment prefix',async()=>{
  const app=await boot('/research/leaderboard','?board=imu&tab=standard&dataset=opportunity_a');
  app.get('leaderboard-wrap').dispatch('click',{target:{closest:s=>s==='[data-result]'?{dataset:{result:'raw-v2:tinyhar:opportunity_a'}}:null}});
  assert.match(app.get('detail-body').innerHTML,/https:\/\/official\.example\/research\/leaderboard\/data\/opportunity_audit_20261030\.json/);
});
test('paper reference, cross-dataset and Sensor QA remain usable under the prefix',async()=>{
  for(const search of ['?board=imu&view=overview','?board=imu&tab=cross-dataset','?board=qa&view=overall']) {
    const app=await boot('/leaderboard',search);
    assert.doesNotMatch(app.get('leaderboard-wrap').innerHTML,/Results could not be loaded|No matching results/);
    assert.match(app.get('leaderboard-wrap').innerHTML,/source-paper|overall-complete|reference-row/);
  }
});
test('Pages submissions use the campus server portal instead of a dead local API',()=>{
 const html=read('submit-model.html');
 assert.match(html,/http:\/\/10\.130\.151\.16:40677\/submit-model\.html/);
 assert.match(html,/campus network or VPN/);
 assert.doesNotMatch(html,/<form|api\/submissions/);
});
test('all dataset downloads point to official HTTPS sources',()=>{
 const data=JSON.parse(read('data/datasets.json'));
 const html=read('datasets.html');
 assert.equal(data.datasets.length,5);
 for(const entry of data.datasets){
  assert.equal(entry.archive_url,entry.official_archive_url);
  assert.match(entry.archive_url,/^https:/);
  assert.ok(html.includes('href="'+entry.official_archive_url+'"'));
 }
 assert.match(html,/checksums describe the mirror snapshot/);
});
test('the homepage and every Overall URL rank all ten methods with visible asterisk notes',async()=>{
 for(const query of ['', '?board=imu&view=overall', '?board=imu&tab=cross-dataset&view=overall&aggregate=onehar-lodo', '?board=imu&view=overall&tab=standard']) {
  const app=await boot('/leaderboard',query);
  const html=app.get('leaderboard-wrap').innerHTML;
  assert.match(app.get('result-summary').textContent,/10 ranked models/);
  for(const id of ['mosaic','onehar','robusthar','unimts','tinyhar','limu_bert','attend_and_discriminate','if_convtransformer','deepsense','deepconvlstm']) assert.ok(html.includes('data-method="'+id+'"'));
  assert.match(html,/92\.22/);
  assert.match(html,/68\.09/);
  assert.match(html,/comparison-asterisk/);
  assert.doesNotMatch(html,/Partial coverage|unranked/);
  assert.equal(app.get('reference-footnote').hidden,false);
  assert.equal(app.get('mode-controls').hidden,true);
 }
 assert.match(read('index.html'),/缺失项不记 0/);
 assert.match(read('index.html'),/训练\/测试划分/);
 assert.match(read('index.html'),/oneHAR 采用论文 supervised joint training/);
});
test('dataset details expose every included score and its original setting',async()=>{
 const app=await boot('/leaderboard');
 app.get('leaderboard-wrap').dispatch('click',{target:{closest:s=>s==='[data-summary]'?{dataset:{summary:'onehar'}}:null}});
 assert.match(app.get('detail-context').textContent,/5 datasets/);
 assert.match(app.get('detail-body').innerHTML,/supervised joint training/);
 assert.match(app.get('detail-body').innerHTML,/91\.90/);
 app.get('detail-body').dispatch('click',{target:{closest:()=>({dataset:{result:'transfer:onehar-joint:onehar:uci_har_6c'}})}});
 assert.match(app.get('detail-body').innerHTML,/Table 2/);
});
test('available-only means are calculated transparently, without coverage priority or selecting the highest value',async()=>{
 const app=await boot('/leaderboard');
 const M=app.env.IoTBenchRanking;
 const fixture=(id,dataset,value,source='run',extra={})=>({id:id+dataset+source,method:{id,name:id},dataset,protocol:'raw-v2',metric:'accuracy',metrics:{accuracy:value},value,source,unit:'percent',direction:'higher',...extra});
 const rows=[fixture('x','wisdm',.99),fixture('x','wisdm_18c',.5,'paper',{paperRole:'original',protocol:'paper'}),fixture('x','hhar',.8),fixture('y','uci_har',.95)];
 const before=JSON.stringify(rows);
 const result=M.referenceRanking(rows,'accuracy','best',['x','y']);
 assert.equal(result[0].method.id,'y');
 assert.equal(result[0].coverage,1);
 assert.equal(result[1].coverage,2);
 assert.equal(result[1].mean,.65);
 assert.equal(JSON.stringify(rows),before);
 const reg=JSON.parse(read('api/results.json')),evidence=JSON.parse(read('data/imu_evidence.json')),repro=JSON.parse(read('data/imu_reproduction_20261028.json')),transfer=JSON.parse(read('data/paper_transfer_evidence.json'));
 const actual=M.referenceRanking([...M.imuObservations(reg,evidence,repro),...M.mainPaperObservations(transfer)]);
 const one=actual.find(e=>e.method.id==='onehar'),mosaic=actual.find(e=>e.method.id==='mosaic');
 assert.equal(one.coverage,5);assert.ok(Math.abs(one.mean-.9222)<1e-12);
 assert.ok(one.values.every(r=>r.protocol==='onehar-joint'));
 assert.equal(mosaic.coverage,18);assert.ok(mosaic.values.every(r=>r.protocol==='mosaic-full-shot'));
 assert.ok(actual.every((e,i)=>i===0||actual[i-1].mean>=e.mean));
 app.get('sort-results').dispatch('click');
 const html=app.get('leaderboard-wrap').innerHTML;
 assert.ok(html.indexOf('data-method="mosaic"')<html.indexOf('data-method="robusthar"'));
});
