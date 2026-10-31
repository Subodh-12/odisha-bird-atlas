/*
Upload page integration.

Responsibilities:
1. Validate the selected image.
2. Display the image preview.
3. Check backend health.
4. Send image to FastAPI.
5. Store the result and photograph.
6. Open the result page.
*/

const API_BASE_URL = (
    window.BIRD_API_BASE_URL ||
    "http://127.0.0.1:8000"
).replace(/\/+$/, "");

const MAX_FILE_SIZE = 10 * 1024 * 1024;

const ALLOWED_TYPES = new Set([
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp"
]);

const DB_NAME = "indian-bird-atlas";

const DB_VERSION = 1;

const STORE_NAME = "current-upload";


const el = {

    uploadArea: document.getElementById("uploadArea"),

    imageInput: document.getElementById("imageInput"),

    previewContainer: document.getElementById("previewContainer"),

    previewImage: document.getElementById("previewImage"),

    previewName: document.getElementById("previewName"),

    fileInfo: document.getElementById("fileInfo"),

    identifyBtn: document.getElementById("identifyBtn"),

    removeImageBtn: document.getElementById("removeImageBtn"),

    loading: document.getElementById("loading"),

    loadingTitle: document.getElementById("loadingTitle"),

    errorMessage: document.getElementById("errorMessage"),

    apiStatus: document.getElementById("apiStatus"),

    apiStatusText: document.getElementById("apiStatusText")

};


let selectedFile = null;

let previewUrl = null;

let isSubmitting = false;


// ============================================================
// INDEXED DB
// ============================================================

function openImageDb() {

    return new Promise((resolve, reject) => {

        const request = indexedDB.open(
            DB_NAME,
            DB_VERSION
        );

        request.onupgradeneeded = () => {

            const db = request.result;

            if (!db.objectStoreNames.contains(STORE_NAME)) {

                db.createObjectStore(STORE_NAME);

            }

        };

        request.onsuccess = () => {

            resolve(request.result);

        };

        request.onerror = () => {

            reject(
                request.error ||
                new Error("Could not open image storage.")
            );

        };

    });

}


async function saveImageForResults(file) {

    const db = await openImageDb();

    try {

        await new Promise((resolve, reject) => {

            const tx = db.transaction(
                STORE_NAME,
                "readwrite"
            );

            tx.objectStore(STORE_NAME).put(
                file,
                "latest"
            );

            tx.oncomplete = resolve;

            tx.onerror = () => {

                reject(
                    tx.error ||
                    new Error("Could not save the photograph.")
                );

            };

            tx.onabort = () => {

                reject(
                    tx.error ||
                    new Error("Image storage was interrupted.")
                );

            };

        });

    } finally {

        db.close();

    }

}


// ============================================================
// ERROR HANDLING
// ============================================================

function showError(message) {

    el.errorMessage.textContent = message;

    el.errorMessage.classList.add("visible");

}


function hideError() {

    el.errorMessage.textContent = "";

    el.errorMessage.classList.remove("visible");

}


// ============================================================
// CHECK BACKEND HEALTH
// ============================================================

async function checkApiHealth() {

    try {

        const response = await fetch(
            `${API_BASE_URL}/health`,
            {
                cache: "no-store"
            }
        );

        if (!response.ok) {

            throw new Error("Backend health check failed.");

        }

        const health = await response.json();

        const ready = health.model_loaded === true;

        el.apiStatus.dataset.state = (
            ready ? "online" : "checking"
        );

        if (!ready) {
            el.apiStatusText.textContent =
                "API online · WingCLIP model is loading";
            return;
        }

        const researchState = health.gemini_configured
            ? "Gemini ready"
            : "Gemini not configured";

        el.apiStatusText.textContent =
            `WingCLIP ready · ${health.device} · ${researchState}`;

    } catch (error) {

        el.apiStatus.dataset.state = "offline";

        el.apiStatusText.textContent = (
            "Identification service offline"
        );

    }

}


// ============================================================
// IMAGE VALIDATION
// ============================================================

function handleFile(file) {

    hideError();

    if (!file) return;

    
    const extension = file.name.split(".").pop().toLowerCase();

    const allowedExtensions = new Set([
        "jpg",
        "jpeg",
        "png",
        "webp",
        "bmp"
    ]);

    const typeIsAllowed = ALLOWED_TYPES.has(file.type);
    const extensionIsAllowed = allowedExtensions.has(extension);

    // Some desktop browsers provide an empty MIME type for valid images.
    // The backend still checks the actual image format.
    if (!typeIsAllowed && !extensionIsAllowed) {
        showError(
            "Please choose a JPEG, PNG, WEBP or BMP image."
        );
        return;
    }

    if (!file.size) {

        showError(
            "The selected file is empty."
        );

        return;

    }

    if (file.size > MAX_FILE_SIZE) {

        showError(
            "The image must not exceed 10 MB."
        );

        return;

    }

    if (previewUrl) {

        URL.revokeObjectURL(previewUrl);

    }

    selectedFile = file;

    previewUrl = URL.createObjectURL(file);

    el.previewImage.src = previewUrl;

    el.previewName.textContent = file.name;

    el.fileInfo.textContent =
        `${(file.size / 1024).toFixed(1)} KB`;

    el.previewContainer.hidden = false;

    el.identifyBtn.disabled = false;

}


// ============================================================
// CLEAR IMAGE
// ============================================================

function clearSelection() {

    selectedFile = null;

    el.imageInput.value = "";

    if (previewUrl) {

        URL.revokeObjectURL(previewUrl);

    }

    previewUrl = null;

    el.previewImage.removeAttribute("src");

    el.previewContainer.hidden = true;

    el.identifyBtn.disabled = true;

    hideError();

}


// ============================================================
// SEND IMAGE TO BACKEND
// ============================================================

async function identifyBird() {

    if (!selectedFile || isSubmitting) {

        return;

    }

    isSubmitting = true;

    hideError();

    el.identifyBtn.disabled = true;

    el.removeImageBtn.disabled = true;

    el.loading.hidden = false;

    el.loadingTitle.textContent =
        "WingCLIP is identifying the bird; Gemini will prepare its species profile...";


    const formData = new FormData();

    formData.append(
        "file",
        selectedFile,
        selectedFile.name
    );


    const controller = new AbortController();

    const timeout = window.setTimeout(
        () => controller.abort(),
        180000
    );


    try {

        const response = await fetch(
            `${API_BASE_URL}/predict`,
            {
                method: "POST",
                body: formData,
                signal: controller.signal
            }
        );


        const responseText = await response.text();

        let payload;

        try {

            payload = responseText
                ? JSON.parse(responseText)
                : {};

        } catch {

            throw new Error(
                `The server returned an invalid response. HTTP ${response.status}`
            );

        }


        if (!response.ok) {

            const detail = (
                typeof payload.detail === "string"
                ? payload.detail
                : `Identification failed. HTTP ${response.status}`
            );

            throw new Error(detail);

        }


        if (
            !payload.identification ||
            !payload.bird_information
        ) {

            throw new Error(
                "The API response is missing bird identification information. Check that the updated backend is running."
            );

        }


        // Save the uploaded image.

        try {
            await saveImageForResults(selectedFile);
        } catch (storageError) {
            console.warn(
                "Could not save uploaded image for the result page:",
                storageError
            );
        }


        // Save the API response.

        sessionStorage.setItem(
            "birdIdentificationResult",
            JSON.stringify(payload)
        );


        // Open the result page.

        window.location.href = "result.html";


    } catch (error) {

        if (error.name === "AbortError") {

            showError(
                "The identification request took too long. Check the backend terminal and try again."
            );

        } else {

            showError(
                error.message ||
                "Something went wrong during bird identification."
            );

        }

    } finally {

        window.clearTimeout(timeout);

        isSubmitting = false;

        el.loading.hidden = true;

        el.identifyBtn.disabled = !selectedFile;

        el.removeImageBtn.disabled = false;

    }

}


// ============================================================
// EVENT LISTENERS
// ============================================================

// Open file picker.

el.uploadArea.addEventListener(
    "click",
    (event) => {

        if (!event.target.closest("button")) {

            el.imageInput.click();

        }

    }
);


// Keyboard support.

el.uploadArea.addEventListener(
    "keydown",
    (event) => {

        if (
            event.key === "Enter" ||
            event.key === " "
        ) {

            event.preventDefault();

            el.imageInput.click();

        }

    }
);


// File selection.

el.imageInput.addEventListener(
    "change",
    (event) => {

        handleFile(
            event.target.files[0]
        );

    }
);


// Remove selected image.

el.removeImageBtn.addEventListener(
    "click",
    clearSelection
);


// Identify button.

el.identifyBtn.addEventListener(
    "click",
    identifyBird
);


// Drag and drop.

["dragenter", "dragover"].forEach(
    (eventName) => {

        el.uploadArea.addEventListener(
            eventName,
            (event) => {

                event.preventDefault();

                el.uploadArea.classList.add("dragover");

            }
        );

    }
);


["dragleave", "drop"].forEach(
    (eventName) => {

        el.uploadArea.addEventListener(
            eventName,
            (event) => {

                event.preventDefault();

                el.uploadArea.classList.remove("dragover");

            }
        );

    }
);


el.uploadArea.addEventListener(
    "drop",
    (event) => {

        const file = (
            event.dataTransfer &&
            event.dataTransfer.files[0]
        );

        if (file) {

            handleFile(file);

        }

    }
);


// Initial API health check.

checkApiHealth();