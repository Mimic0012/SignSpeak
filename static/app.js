const webcam = document.getElementById("webcam");
const predictionEl = document.getElementById("prediction");
const translationEl = document.getElementById("translation");
const confidenceFill = document.getElementById("confidence-fill");
const confidenceText = document.getElementById("confidence-text");
const sentenceEl = document.getElementById("sentence");
const clearBtn = document.getElementById("clearSentenceBtn");
const translateBtn = document.getElementById("translateBtn");

let lastWord = "";
let sentenceWords = [];

clearBtn.onclick = () => {
    sentenceWords = [];
    sentenceEl.textContent = "---";
    translationEl.textContent = "---";
};

translateBtn.onclick = async () => {
    const sentenceText = sentenceWords.join(" ");
    if (!sentenceText) return;
    const res = await fetch("http://localhost:5000/translate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sentence: sentenceText, lang: "hi" })
    });
    const { translation } = await res.json();
    translationEl.textContent = translation;
    translationEl.classList.remove("show");
    void translationEl.offsetWidth;
    translationEl.classList.add("show");
};

document.getElementById("speakHindiBtn").onclick = async () => {
    const sentenceText = sentenceWords.join(" ");
    if (!sentenceText) return;

    const res = await fetch("http://localhost:5000/translate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sentence: sentenceText, lang: "hi" })
    });
    const { translation } = await res.json();
    speak(translation, "hi");
};

document.getElementById("speakEnglishBtn").onclick = () => {
    const sentenceText = sentenceWords.join(" ");
    if (!sentenceText) return;
    speak(sentenceText, "en");
};

function speak(text, lang) {
    fetch("http://localhost:5000/speak", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, lang })
    })
        .then(r => r.blob())
        .then(b => {
            const url = URL.createObjectURL(b);
            new Audio(url).play();
        });
}

async function initWebcam() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    webcam.srcObject = stream;
}

function captureFrameBlob() {
    const canvas = document.createElement("canvas");
    canvas.width = webcam.videoWidth;
    canvas.height = webcam.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(webcam, 0, 0, canvas.width, canvas.height);
    return new Promise(res => canvas.toBlob(res, "image/jpeg"));
}

async function recognizeFrame() {
    const blob = await captureFrameBlob();
    const fd = new FormData();
    fd.append("image", blob, "frame.jpg");

    const res = await fetch("http://localhost:5000/recognize", { method: "POST", body: fd });
    const data = await res.json();
    const word = data.prediction;
    const confidence = data.confidence;
    const confidencePct = (confidence * 100).toFixed(1);

    if (confidence >= 0.75 && word && word !== lastWord) {
        lastWord = word;
        predictionEl.textContent = word;
        confidenceFill.style.width = confidencePct + "%";
        confidenceText.textContent = confidencePct + "%";

        sentenceWords.push(word);
        sentenceEl.textContent = sentenceWords.join(" ");
    } else {
        predictionEl.textContent = "Adjust your hands properly";
        confidenceFill.style.width = confidencePct + "%";
        confidenceText.textContent = confidencePct + "%";
    }
}

initWebcam().then(() => {
    setInterval(recognizeFrame, 800);
});