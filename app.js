/* Browser inference using the exact Keras-trained LSTM and dense weights. */
const names=['Toxic','Severe toxic','Obscene','Threat','Insult','Identity hate'];
let params={},dictionary,ready=false;
const status=document.getElementById('status'),button=document.getElementById('analyze');
async function init(){
 try{
  await tf.ready();
  const [meta,vocab,binary]=await Promise.all([fetch('weights.json').then(r=>r.json()),fetch('vocabulary.json').then(r=>r.json()),fetch('weights.bin').then(r=>r.arrayBuffer())]);
  const floats=new Float32Array(binary); for(const [name,item] of Object.entries(meta)){params[name]=tf.tensor(floats.slice(item.offset/4,item.offset/4+item.length),item.shape);}
  dictionary=new Map(vocab.map((word,index)=>[word,index]));
  ready=true;button.disabled=false;button.textContent='Analyze comment';status.textContent='Model ready · runs privately in your browser';
 }catch(err){console.error(err);status.textContent='Model failed to load: '+err.message+' (Please use a local web server or hosted website.)';}
}
// Keras TextVectorization default: lowercase + remove ASCII punctuation; whitespace tokenization; post-pad to 300.
function tokenize(s){
 const clean=s.toLowerCase().replace(/[!"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~]/g,'');
 const tokens=clean.trim()?clean.trim().split(/\s+/):[];
 const ids=tokens.slice(0,300).map(w=>dictionary.get(w)??1);
 while(ids.length<300)ids.push(0);
 return ids;
}
function lstm(sequence,side,reverse){
 const kernel=params[side+'_kernel'],rec=params[side+'_recurrent'],bias=params[side+'_bias'];
 let h=tf.zeros([1,32]),c=tf.zeros([1,32]);
 for(let step=0;step<300;step++){
  const t=reverse?299-step:step;
  const [nextH,nextC]=tf.tidy(()=>{
   const x=sequence.slice([t,0],[1,32]);
   const gates=x.matMul(kernel).add(h.matMul(rec)).add(bias);
   const [i,f,g,o]=tf.split(gates,4,1);
   const nc=tf.sigmoid(f).mul(c).add(tf.sigmoid(i).mul(tf.tanh(g)));
   const nh=tf.sigmoid(o).mul(tf.tanh(nc));
   return [nh,nc];
  });
  h.dispose();c.dispose();h=nextH;c=nextC;
 }
 c.dispose();return h;
}
function predict(ids){
 const seq=tf.tidy(()=>tf.gather(params.embedding,tf.tensor1d(ids,'int32')));
 const forward=lstm(seq,'forward',false),backward=lstm(seq,'backward',true);
 const out=tf.tidy(()=>{
  let x=tf.concat([forward,backward],1);
  for(let n=0;n<4;n++){
   x=x.matMul(params['d'+n+'_kernel']).add(params['d'+n+'_bias']);
   x=n===3?tf.sigmoid(x):tf.relu(x);
  }
  return x;
 });
 const scores=Array.from(out.dataSync());seq.dispose();forward.dispose();backward.dispose();out.dispose();return scores;
}
button.addEventListener('click',async()=>{
 if(!ready)return;const comment=document.getElementById('comment').value.trim();if(!comment){status.textContent='Please type a comment first.';return;}
 button.disabled=true;button.textContent='Analyzing…';status.textContent='Running your trained neural network…';
 await new Promise(r=>setTimeout(r,40));
 try{
  const scores=predict(tokenize(comment));
  document.getElementById('metrics').innerHTML=names.map((name,i)=>`<div class="metric"><div class="metric-head"><span>${name}</span><strong>${(scores[i]*100).toFixed(1)}%</strong></div><div class="track"><div class="fill" style="width:${Math.max(0,Math.min(100,scores[i]*100))}%"></div></div></div>`).join('');
  document.getElementById('results').style.display='block';status.textContent='Analysis complete · '+(scores.some(v=>v>=.3)?'One or more categories above 30%':'No categories above 30%');
 }catch(e){console.error(e);status.textContent='Prediction failed: '+e.message;}
 button.disabled=false;button.textContent='Analyze comment';
});
init();
