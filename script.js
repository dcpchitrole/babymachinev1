tf.setBackend('cpu');
console.log("Using backend:", tf.getBackend());
window.addEventListener('load', async () => {
    datasets.A = [];
    datasets.B = [];
    datasets.C = [];
    document.getElementById("capture").textContent = "Captured Images: 0";
    showDatasetButton.style.display = "none";
    clearDatasetButton.style.display = "none";
    try {
        const models = await tf.io.listModels();
        for (const modelName in models) {
            await tf.io.removeModel(modelName);
        }
        console.log("Cleared cached models");
    } catch (error) {
        console.log("No cached models found");
    }
});
const startCameraButton = document.getElementById("start-camera");
const captureImageButton = document.getElementById("capture-image");
const videoElement = document.getElementById("video");
const canvasElement = document.getElementById("canvas");
const datasetClassSelect = document.getElementById("dataset-class");
const showDatasetButton = document.getElementById("show-dataset");
const clearDatasetButton = document.getElementById("clear-dataset");
const imageCountDisplay = document.createElement("p");
document.body.appendChild(imageCountDisplay);
const datasets = {
    A: [],
    B: [],
    C: []
};
let cameraStream = null;
let capturingInterval = null;
let model = null;
let isTraining = false;
let isTesting = false;
let animationFrameId = null;
startCameraButton.addEventListener("click", () => {
    cameraStream ? stopCamera() : startCamera();
});
function startCamera() {
    navigator.mediaDevices.getUserMedia({ video: true })
        .then((stream) => {
            cameraStream = stream;
            videoElement.srcObject = stream;
            videoElement.style.display = "block";
            canvasElement.style.display = "none";
            startCameraButton.textContent = "Stop Camera";
        })
        .catch((error) => {
            console.error("Camera error:", error);
            alert("Camera access denied. Please enable permissions.");
        });
}
function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
        videoElement.style.display = "none";
        startCameraButton.textContent = "Start Camera";
    }
}
captureImageButton.addEventListener("mousedown", () => cameraStream && startContinuousCapture());
captureImageButton.addEventListener("mouseup", stopContinuousCapture);
captureImageButton.addEventListener("mouseleave", stopContinuousCapture);
function startContinuousCapture() {
    capturingInterval = setInterval(captureImage, 100);
}
function stopContinuousCapture() {
    clearInterval(capturingInterval);
    capturingInterval = null;
}
function captureImage() {
    const context = canvasElement.getContext("2d");
    canvasElement.width = videoElement.videoWidth;
    canvasElement.height = videoElement.videoHeight;
    context.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);

    canvasElement.toBlob((blob) => {
        const selectedClass = datasetClassSelect.value;
        datasets[selectedClass].push(blob);

        console.log(`Captured image for Class ${selectedClass}:`, blob);

        document.getElementById("capture").textContent =
            `Captured Images in Class ${selectedClass}: ${datasets[selectedClass].length}`;

        showDatasetButton.style.display = "inline-block";
        clearDatasetButton.style.display = "inline-block";
    }, 'image/jpeg', 0.80); // Adjust quality as needed
}
showDatasetButton.addEventListener("click", () => {
    const selectedClass = datasetClassSelect.value;
    const images = datasets[selectedClass];

    if (!images || images.length === 0) {
        alert(`No images found for Class ${selectedClass}!`);
        return;
    }

    console.log(`Showing dataset for Class ${selectedClass}:`, images);

    const newTab = window.open();
    newTab.document.write('<html><head><title>Captured Images</title></head><body>');
    newTab.document.write(`<h1>Dataset: Class ${selectedClass}</h1>`);
    newTab.document.write('<div style="display: flex; flex-wrap: wrap;">');

    images.forEach((blob, index) => {
        try {
            const url = URL.createObjectURL(blob);
            newTab.document.write(`<div style="margin: 10px; width: 200px;">
                <img src="${url}" style="max-width: 100%; border-radius: 8px;">
                <p>Image ${index + 1}</p>
            </div>`);
        } catch (error) {
            console.error("Error creating object URL:", error);
        }
    });

    newTab.document.write('</div></body></html>');
});
clearDatasetButton.addEventListener("click", () => {
    const selectedClass = datasetClassSelect.value;
    datasets[selectedClass].length = 0;
    document.getElementById("capture").textContent = `Captured Images in Class ${selectedClass}: 0`;
    alert(`Dataset for Class ${selectedClass} cleared successfully!`);
});
function resetModel() {
    if (model) {
        model.dispose();
        tf.disposeVariables();
        model = null;
        console.log("♻️ Model reset complete");
    }
}
async function createModel() {
    console.log("🔄 Creating new model...");
    const model = tf.sequential({
        layers: [
            tf.layers.flatten({ inputShape: [224, 224, 3] }),
            tf.layers.dense({ units: 128, activation: 'relu' }), 
            tf.layers.dense({ units: 3, activation: 'softmax' })
        ]
    });
    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });
    return model;
}
document.getElementById('train-model').addEventListener('click', async () => {
    if (isTraining) return;
    if (isAnyDatasetEmpty()) {
        alert("Capture data for at least one class first!");
        return;
    }
    isTraining = true;
    try {
        resetModel();
        model = await createModel();
        const { xs, ys } = await prepareTrainingData();
        console.log("Starting training...");
        console.log("Input tensors:", xs.shape);
        console.log("Labels:", ys.shape);
        await model.fit(xs, ys, {
            epochs: 20,
            batchSize: 8,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    const progress = ((epoch + 1) / 20) * 100;
                    document.getElementById('progress-bar').value = progress;
                    document.getElementById('progress-text').textContent = 
                        `${Math.round(progress)}% (Loss: ${logs.loss.toFixed(2)})`;
                    console.log(`Epoch ${epoch + 1}: Loss = ${logs.loss.toFixed(4)}`);
                }
            }
        });
        console.log("✅ Training complete!");
        alert("Training complete!");
    } catch (error) {
        console.error("Training failed:", error);
        alert(`Training error: ${error.message}`);
    } finally {
        isTraining = false;
    }
});
async function prepareTrainingData() {
    const xs = []; 
    const ys = [];

    await Promise.all(
        Object.entries(datasets).map(async ([cls, blobs], idx) => {
            for (const blob of blobs) {
                const tensor = await blobToTensor(blob);
                xs.push(tensor);
                ys.push(idx);
            }
        })
    );
    console.log("Training data prepared:");
    console.log("Input tensors:", xs.length);
    console.log("Labels:", ys);

    return {
        xs: tf.concat(xs),
        ys: tf.oneHot(tf.tensor1d(ys, 'int32'), 3).cast('float32')
    };
}
document.getElementById('test-model').addEventListener('click', async () => {
    if (!model) {
        alert("Train a model first!");
        return;
    }
    isTesting = !isTesting;
    const testButton = document.getElementById('test-model');
    testButton.textContent = isTesting ? "Stop Testing" : "Start Testing";
    if (isTesting) {
const predictFrame = async () => {
    if (!isTesting) return;
    
    const context = canvasElement.getContext("2d");
    canvasElement.width = videoElement.videoWidth;
    canvasElement.height = videoElement.videoHeight;
    context.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
    const tensor = tf.browser.fromPixels(canvasElement)
        .resizeNearestNeighbor([224, 224])
        .toFloat()
        .div(255.0)
        .expandDims();
    try {
        const prediction = model.predict(tensor);
        const probs = await prediction.data();
        document.querySelectorAll('.confidence-bar').forEach((bar, idx) => {
            const percent = (probs[idx] * 100).toFixed(1);
            bar.querySelector('progress').value = probs[idx];
            bar.querySelector('.confidence-value').textContent = `${percent}%`;
        });
        tensor.dispose();
        prediction.dispose();
    } catch (error) {
        console.error("Prediction error:", error);
    }
    
    animationFrameId = requestAnimationFrame(predictFrame);
};
        predictFrame();
    } else {
        cancelAnimationFrame(animationFrameId);
    }
});
document.getElementById('export-model').addEventListener('click', async () => {
    if (!model) {
        alert("No model to export!");
        return;
    }

    try {
        tf.engine().startScope();
        const modelName = `baby-machine-${Date.now()}`;
        await model.save(`downloads://${modelName}`);
        tf.engine().endScope();
        console.log("💾 Model exported:", modelName);
        alert(`Model exported as ${modelName}`);
    } catch (error) {
        console.error("Export failed:", error);
        alert("Export failed! Check console.");
    }
});
function isAnyDatasetEmpty() {
    return Object.values(datasets).every(arr => arr.length === 0);
}

async function blobToTensor(blob) {
    return new Promise((resolve) => {
        const img = new Image();
        img.src = URL.createObjectURL(blob);
        img.onload = () => {
            const tensor = tf.browser.fromPixels(img)
                .resizeNearestNeighbor([224, 224])
                .toFloat()
                .div(255.0)
                .expandDims();
            URL.revokeObjectURL(img.src);
            resolve(tensor);
        };
    });
}
async function logModelWeights() {
    if (!model) {
        console.log("No model available.");
        return;
    }

    const weights = await model.getWeights();
    console.log("Model weights:");
    weights.forEach((w, i) => {
        console.log(`Weight ${i}:`, w.dataSync());
    });
}