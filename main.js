function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}
let model = null;
let loadModel = async function(){
  model = await tf.loadLayersModel('https://raw.githubusercontent.com/kubzoey95/bach/main/model.json');
  model.resetStates();
  console.log("Model loaded!");
  console.log(model);
}

loadModel()

let synth = 0;

let loadSynth = async function(){
  
  synth = new Tone.Sampler({
	urls: {
		A2: "A2.mp3",
		A4: "A4.mp3",
		A6: "A6.mp3",
		B1: "B1.mp3",
		B3: "B3.mp3",
		B5: "B5.mp3",
		B6: "B6.mp3",
		C3: "C3.mp3",
		C5: "C5.mp3",
		D2: "D2.mp3",
		D4: "D4.mp3",
		D6: "D6.mp3",
		D7: "D7.mp3",
		E1: "E1.mp3",
		E3: "E3.mp3",
		E5: "E5.mp3",
		F2: "F2.mp3",
		F4: "F4.mp3",
		F6: "F6.mp3",
		F7: "F7.mp3",
		G1: "G1.mp3",
		G3: "G3.mp3",
		G5: "G5.mp3"
	},
	  attack: 1,
	baseUrl: "https://raw.githubusercontent.com/nbrosowsky/tonejs-instruments/master/samples/harp/"
}).toDestination()
  synth.volume.value = -20;
  console.log("Synth loaded!");
  console.log(synth);
}

loadSynth()

let noteStack = [];

let lastTime = performance.now();

let refreshTime = 0;

const canvas = document.querySelector('canvas');
const engine = new BABYLON.Engine(canvas, true);

let timeDelta = 0;
let time = performance.now();
let catmullRom = null;
let catmullRomSpline = null;
let path = []
for (let i=0;i<20;i++){
	path.push(BABYLON.Vector3.Zero());
}
let camera = null;
const createScene = function () {
	const scene = new BABYLON.Scene(engine);
	scene.clearColor = new BABYLON.Color3(0, 0, 0);
	camera = new BABYLON.ArcRotateCamera('camera', Math.PI / 2, 0, 100, new BABYLON.Vector3(0, 0, 0), scene);
    	camera.mode = BABYLON.Camera.ORTHOGRAPHIC_CAMERA;
	catmullRomSpline = BABYLON.Mesh.CreateLines(null, path, null, null, catmullRomSpline);
	return scene;
};

const scene = createScene();

var updatePath = function(path) {
	    for (var i = 0; noteStack.length > 1 && i < noteStack.length && i < path.length; i++) {
	      var x = noteStack[noteStack.length - 1 - i].x;
	      var z = noteStack[noteStack.length - 1 - i].z;
	      var y = noteStack[noteStack.length - 1 - i].y;
	      path[i].x = x;
	      path[i].y = y;
	      path[i].z = z;
	    }
};
let render = function(){
	catmullRomSpline = BABYLON.Mesh.CreateLines("catmullRomSpline", path, scene, true);
	scene.registerBeforeRender(function() {
	    updatePath(path);
	    catmullRomSpline = BABYLON.Mesh.CreateLines(null, path, null, null, catmullRomSpline);
	});
	engine.runRenderLoop(function () {
		let perf = performance.now();
		timeDelta = perf - time;
		time = perf;
		for (let note of noteStack){
			note.x += timeDelta / 10;
		}
		scene.render();
});

window.addEventListener("resize", function () {
	engine.resize();
});
}



let toneStarted = false;
const now = Tone.now();

let currentTone = null;

let currentChord = null;
let lastNotes = [0,0,0];

let currentTime = null;
let lastTimes = [0,0,0];

let playAndPush = async function(toneToPlay, time=0.25){
  synth && synth.triggerAttackRelease(Math.pow(2, (toneToPlay + 3) / 12) * 440.0, 5, Tone.now());
  noteStack.push(new BABYLON.Vector3(-canvas.getBoundingClientRect().width / 2, 0, -(toneToPlay - 5) / 25 * canvas.getBoundingClientRect().height / 2));
  if (noteStack.length > path.length){
  	noteStack = noteStack.slice(noteStack.length - path.length)
  }
  await sleep(time * 1000);
}

let chooseRandomNumber = function(weights, lnght = 4){
  let sum = 0;
  let weightsEntries = Array.from(weights.entries());
  weightsEntries.sort((e1, e2) => e1[1] - e2[1]);
  weightsEntries = weightsEntries.slice(weightsEntries.length - lnght);
  weightsEntries = weightsEntries.map((e) => [e[0], (e[1] - weightsEntries[0][1]) / (weightsEntries[weightsEntries.length - 1][1] - weightsEntries[0][1])]);
  let randomNumber = Math.random();
  for(let [index, weight] of weightsEntries){
    let newSum = sum + weight;
    if (randomNumber < newSum){
      return index;
    }
    sum = newSum;
  }
}

let firstRun = true;

let goThroughModel = function(){
  let prediction = null;
  while(lastNotes.length > 2){
    let lastNotesTensor = tf.oneHot(tf.tensor2d([lastNotes.slice(0,3)], [1, 3], 'int32'), 26);
    let lastTimesTensor = tf.tensor2d([lastTimes.slice(0,3)], [1, 3], 'float32').reshape([1, 3, 1]);
    prediction = model.predict([lastNotesTensor, lastTimesTensor]);
    lastNotes = lastNotes.slice(1);
    lastTimes = lastTimes.slice(1);
  }
  prediction && lastNotes.push(chooseRandomNumber(Array.from(prediction[0].reshape([26]).dataSync()), firstRun ? 25 : 4));
  prediction && lastTimes.push(prediction[1].dataSync()[0])
  firstRun = false;
}

let predictMelody = async function(){
    if (lastNotes.length > 2){
      goThroughModel();
    }
    currentTone += lastNotes[lastNotes.length - 1] - 1 - 12;
    if (currentTone > 30){
      currentTone -= 12;
    }
    if (currentTone < -20){
      currentTone += 12;
    }
    await playAndPush(currentTone, lastTimes[lastTimes.length - 1]);
}

let play = false;
render();
let playLoop = async function(){
	while(!play || !synth.loaded){
		await sleep(500);
	}
	while(play){
		await predictMelody();
	}
}
playLoop();
scene.onPointerObservable.add(async function(e){
			      if(!toneStarted){
				    await Tone.start();
				    toneStarted = true;
				}
			      play = true;
			      });
