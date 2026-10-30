import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import vm from 'node:vm';

const root=fileURLToPath(new URL('..',import.meta.url));
const read=name=>fs.readFileSync(path.join(root,'site',name),'utf8');
async function boot(prefix,search='?board=imu&view=overall&tab=standard') {
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
    assert.match(app.get('leaderboard-wrap').innerHTML,/source-paper|overall-complete/);
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
