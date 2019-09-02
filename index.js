let net;

const classifier = knnClassifier.create();
const webcamElement = document.getElementById('webcam');
const classes = ['A', 'B', 'C', 'Empty'];

var current_prediction = 0;

async function save() {
   let dataset = this.classifier.getClassifierDataset()
   var datasetObj = {}
   Object.keys(dataset).forEach((key) => {
     let data = dataset[key].dataSync();
     // use Array.from() so when JSON.stringify() it covert to an array string e.g [0.1,-0.2...] 
     // instead of object e.g {0:"0.1", 1:"-0.2"...}
     datasetObj[key] = Array.from(data); 
   });
   let jsonStr = JSON.stringify(datasetObj)
   //can be change to other source
   localStorage.setItem("myData", jsonStr);
 }

async function load() {
     //can be change to other source
    let dataset = localStorage.getItem("myData")
    let tensorObj = JSON.parse(dataset)
    //covert back to tensor
    Object.keys(tensorObj).forEach((key) => {
      tensorObj[key] = tf.tensor(tensorObj[key], [tensorObj[key].length / 1000, 1000])
    })
    this.classifier.setClassifierDataset(tensorObj);
  }


async function app() {
  console.log('Loading mobilenet..');

  // Load the model.
  net = await mobilenet.load();
  console.log('Sucessfully loaded model');

  await setupWebcam();

  // Reads an image from the webcam and associates it with a specific class
  // index.
  const addExample = classId => {
    // Get the intermediate activation of MobileNet 'conv_preds' and pass that
    // to the KNN classifier.
    const activation = net.infer(webcamElement, 'conv_preds');

    // Pass the intermediate activation to the classifier.
    classifier.addExample(activation, classId);
    console.log('Added sample ${classID}');
  };

  // When clicking a button, add an example for that class.
  document.getElementById('class-a').addEventListener('click', () => addExample(0));
  document.getElementById('class-b').addEventListener('click', () => addExample(1));
  document.getElementById('class-c').addEventListener('click', () => addExample(2));
  document.getElementById('class-e').addEventListener('click', () => addExample(3));

  // START -- buttons for model save/load  
  const localLoadButton = document.getElementById('load-local');
  const localSaveButton = document.getElementById('save-local');


  localLoadButton.addEventListener('click', async () => {
   await load();
  });

  localSaveButton.addEventListener('click', async () => {
    await save();
  });
  // END -- buttons for model save/load


  // add initial variables for calulation number of items 
  var numA = 0;
  var numB = 0;
  var numc = 0;

  var prev = 'Empty'
  var is_new = false;

  document.getElementById('previous').innerText = prev;

  while (true) {
    if (classifier.getNumClasses() > 0) {
      // Get the activation from mobilenet from the webcam.
      const activation = net.infer(webcamElement, 'conv_preds');
      // Get the most likely class and confidences from the classifier module.
      const result = await classifier.predictClass(activation);

      current_prediction = result.classIndex;
      
      document.getElementById('console').innerText = `
        prediction: ${classes[result.classIndex]}\n
        probability: ${result.confidences[result.classIndex]}
      `;
    }

    // after 2 empty prediction counter resets and wait for new item 
    if (prev == 'Empty' && classes[current_prediction] == 'Empty'){
     is_new = true;
    }

    if (is_new){
     switch (classes[current_prediction]){
      case 'A':
        numA += 1;
        document.getElementById('Acount').innerText = numA;
        is_new = false;
        break;
      case 'B':
        numB += 1;
        document.getElementById('Bcount').innerText = numB;
        is_new = false;
        break;
      case 'C':
        numC += 1;
        document.getElementById('Ccount').innerText = numC;
        is_new = false;
        break;
      case 'Empty':
        document.getElementById('console').innerText = 'Still waiting...';
        break;
     }

    }


    await tf.nextFrame();
  }
};


async function setupWebcam() {
  return new Promise((resolve, reject) => {
    const navigatorAny = navigator;
    navigator.getUserMedia = navigator.getUserMedia ||
        navigatorAny.webkitGetUserMedia || navigatorAny.mozGetUserMedia ||
        navigatorAny.msGetUserMedia;
    if (navigator.getUserMedia) {
      navigator.getUserMedia({video: true},
        stream => {
          webcamElement.srcObject = stream;
          webcamElement.addEventListener('loadeddata',  () => resolve(), false);
        },
        error => reject());
    } else {
      reject();
    }
  });
};

app();