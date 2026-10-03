import {readFileSync,writeFileSync} from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
const root=new URL('../',import.meta.url);
const html=readFileSync(new URL('index.html',root),'utf8');
const stored=vm.runInNewContext('('+html.match(/const policy=(\[[\s\S]*?\n\]),names/)[1]+')');
const policy=stored.map(([c,g])=>[c,g,1-c-g]);
const original=JSON.parse(readFileSync(new URL('tools/equilibrium.json',root),'utf8'));
const delta=[1,0,-1];
// Evaluate a full-game best response against the exact rounded policy shipped.
// Successors with one energy >=5 are terminal in value. (5,5) -> (0,0)
// by the now-mandatory simultaneous special, without approximation or discount.
const kernels=Array.from({length:25},(_,s)=>{
  const a=Math.floor(s/5),b=s%5,q=policy[b*5+a],columns=b?[0,1,2]:[0,1];
  return (a?[0,1,2]:[0,1]).map(x=>{
    const probabilities=Array(25).fill(0);let reward=0;
    for(const y of columns){
      if(x===2&&y===0){reward+=q[y];continue;}
      if(x===0&&y===2){reward-=q[y];continue;}
      const na=a+delta[x],nb=b+delta[y];
      if(na>=5&&nb<5)reward+=q[y];
      else if(nb>=5&&na<5)reward-=q[y];
      else probabilities[(na%5)*5+nb%5]+=q[y];
    }
    return {x,reward,probabilities};
  });
});
const expected=(action,v)=>action.reward+action.probabilities.reduce((sum,p,s)=>sum+p*v[s],0);
let v=Array(25).fill(0),iterations=0,residual=Infinity;
for(;iterations<10000;iterations++){
  const next=kernels.map(actions=>Math.max(...actions.map(action=>expected(action,v))));
  residual=Math.max(...next.map((value,s)=>Math.abs(value-v[s])));v=next;
  if(residual<1e-14)break;
}
assert.ok(residual<1e-13);
const selected=kernels.map(actions=>actions.reduce((best,action)=>expected(action,v)>expected(best,v)?action:best));
// Evaluate the greedy response by solving the complete absorbing Markov chain.
const linear=selected.map((action,s)=>[...action.probabilities.map((p,t)=>(s===t?1:0)-p),action.reward]);
for(let j=0;j<25;j++){
  let pivot=j;for(let i=j+1;i<25;i++)if(Math.abs(linear[i][j])>Math.abs(linear[pivot][j]))pivot=i;
  assert.ok(Math.abs(linear[pivot][j])>1e-12);
  [linear[j],linear[pivot]]=[linear[pivot],linear[j]];
  const scale=linear[j][j];for(let k=j;k<=25;k++)linear[j][k]/=scale;
  for(let i=0;i<25;i++)if(i!==j){const factor=linear[i][j];for(let k=j;k<=25;k++)linear[i][k]-=factor*linear[j][k];}
}
const solved=linear.map(row=>row[25]);
const optimalityResidual=Math.max(...kernels.map((actions,s)=>Math.abs(Math.max(...actions.map(action=>expected(action,solved)))-solved[s])));
assert.ok(optimalityResidual<1e-12);
const gaps=solved.map((upper,s)=>upper+solved[(s%5)*5+Math.floor(s/5)]);
const gains=solved.map((upper,s)=>upper-original.values[s]);
const roundingError=Math.max(...policy.flatMap((p,s)=>p.map((x,i)=>Math.abs(x-original.strategies[s][i]))));
const report={
  model:'Infinite duration; win +1, loss -1, nontermination 0; no discount, energy cap or turn cutoff.',
  tableStates:25,tableProbabilityDecimalPlaces:7,maxProbabilityRoundingError:roundingError,
  equilibriumIterationDelta:original.residual,bestResponseIterations:iterations,bestResponseIterationDelta:residual,
  fullGameOptimalityResidual:optimalityResidual,
  maxFullGameNashGap:Math.max(...gaps),maxFullGameBestResponseGain:Math.max(...gains),
  initialStateBestResponseGain:solved[0],
  maxWinProbabilityDecreasePercentagePoints:Math.max(...gains)/2*100,
  initialStateWinProbabilityDecreasePercentagePoints:solved[0]/2*100,
  bestResponseValues:solved
};
writeFileSync(new URL('tools/accuracy.json',root),JSON.stringify(report,null,2));
console.log(JSON.stringify(report));
