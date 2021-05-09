const KEY_TONE_MAPPING = {       "2": 1,       "3": 3,               "5": 6,       "6": 8,       "7": 10,         // black keys
                          "q": 0,       "w": 2,       "e": 4, "r": 5,       "t": 7,       "y": 9,        "u": 11} // white keys
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}
let model = null;
let loadModel = async function(){
  model = await tf.loadLayersModel('https://raw.githubusercontent.com/kubzoey95/chorder/main/model.json');
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
//   synth.connect(new Tone.Freeverb({roomSize : 0.9 , dampening : 3000}).toMaster());
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
// 	if (noteStack.length > 0){
// 		for (let note of noteStack){
// 			note.x -= 10 * timeDelta / 1000;
// 		}
// 		var catmullRomSpline = BABYLON.Mesh.CreateLines("catmullRom", catmullRom.getPoints(), scene);
// 	}
	camera = new BABYLON.ArcRotateCamera('camera', Math.PI / 2, 0, 100, new BABYLON.Vector3(0, 0, 0), scene);
    	camera.mode = BABYLON.Camera.ORTHOGRAPHIC_CAMERA;
// 	camera.attachControl(canvas, true);
	catmullRomSpline = BABYLON.Mesh.CreateLines(null, path, null, null, catmullRomSpline);
	return scene;
};

const scene = createScene(); //Call the createScene function

var updatePath = function(path) {
// 	    let meanVect = 0;
// 	    let cnt = 0;
	    for (var i = 0; noteStack.length > 1 && i < noteStack.length && i < path.length; i++) {
// 	      cnt += 1;
	      var x = noteStack[noteStack.length - 1 - i].x;
	      var z = noteStack[noteStack.length - 1 - i].z;
	      var y = noteStack[noteStack.length - 1 - i].y;
	      path[i].x = x;
	      path[i].y = y;
	      path[i].z = z;
// 	      meanVect += z;
	    }
// 	if (cnt > 0){
// 		camera.position.z = camera.position.z + ((meanVect / cnt) - camera.position.z) * timeDelta;
// 	}
};
let render = function(){
	catmullRomSpline = BABYLON.Mesh.CreateLines("catmullRomSpline", path, scene, true);
	scene.registerBeforeRender(function() {
	    updatePath(path);
	    catmullRomSpline = BABYLON.Mesh.CreateLines(null, path, null, null, catmullRomSpline);
	});
	// Register a render loop to repeatedly render the scene
	engine.runRenderLoop(function () {
		let perf = performance.now();
		timeDelta = perf - time;
		time = perf;
		for (let note of noteStack){
			note.x += timeDelta / 10;
		}
		scene.render();
});

// Watch for browser/canvas resize events
window.addEventListener("resize", function () {
	engine.resize();
});
}



let toneStarted = false;
const now = Tone.now();

let currentTone = null;

let currentChord = null;
let lastNotes = [0,0,0];

let playAndPush = function(toneToPlay){
  synth && synth.triggerAttackRelease(Math.pow(2, (toneToPlay + 3) / 12) * 440.0, 5, Tone.now());
  noteStack.push(new BABYLON.Vector3(-canvas.getBoundingClientRect().width / 2, 0, -(toneToPlay - 5) / 25 * canvas.getBoundingClientRect().height / 2));
  if (noteStack.length > path.length){
  	noteStack = noteStack.slice(noteStack.length - path.length)
  }
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
    prediction = model.predict([lastNotesTensor]);
    lastNotes = lastNotes.slice(1);
  }
  prediction && lastNotes.push(chooseRandomNumber(Array.from(prediction.reshape([26]).dataSync()), firstRun ? 25 : 4));
  firstRun = false;
}

// $(document).keypress(async function(e){
//   if(!toneStarted){
//     await Tone.start();
//     toneStarted = true;
//     render();
//   }
//   let keyPressed = String.fromCharCode(e.keyCode || e.which).toLowerCase();
//   if (KEY_TONE_MAPPING.hasOwnProperty(keyPressed) && currentTone !== KEY_TONE_MAPPING[keyPressed]){
//     let diff = KEY_TONE_MAPPING[keyPressed] - currentTone;
//     if (Math.abs(diff) > 12){
//       diff = KEY_TONE_MAPPING[keyPressed] - ((Math.floor(KEY_TONE_MAPPING[keyPressed] / 12) * 12) + (currentTone % 12));
//     }
//     currentTone += diff;
//     playAndPush(currentTone);
//     lastNotes.push(diff + 12 + 1);
//     console.log(lastNotes);
//   }

// })

let predictMelody = function(){
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
    playAndPush(currentTone);
}

let play = false;
render();
let playLoop = async function(){
	while(!play || !synth.loaded){
		await sleep(500);
	}
// 	let firstNote = Math.floor(Math.random() * 24) - 12;
// 	playAndPush(firstNote);
// 	lastNotes.push(firstNote + 12 + 1);
	while(play){
		await sleep(250);
		predictMelody();
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
// $(canvas).mousedown(playLoop);
// $(document).keyup(async function(e){
//   let keyPressed = String.fromCharCode(e.keyCode || e.which).toLowerCase();
//   console.log(keyPressed);
//   if(typeof model === null){
//     return
//   }
//   if (KEY_TONE_MAPPING.hasOwnProperty(keyPressed)){
//   }
//   else {
//     predictMelody();
//   }
// })
