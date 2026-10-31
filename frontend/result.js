/*
Result page renderer.

WingCLIP supplies the identification.
Gemini supplies the species profile.
All API text is inserted using textContent.
*/

const DB_NAME = "indian-bird-atlas";
const DB_VERSION = 1;
const STORE_NAME = "current-upload";
const RESULT_KEY = "birdIdentificationResult";


const byId = (id) => document.getElementById(id);


const fields = {

    error: byId("resultError"),

    image: byId("uploadedBirdImage"),

    filename: byId("uploadedFilename"),

    common: byId("birdCommonName"),

    scientific: byId("birdScientificName"),

    description: byId("birdDescription"),

    habitat: byId("birdHabitat"),

    diet: byId("birdDiet"),

    distribution: byId("birdDistribution"),

    family: byId("birdFamily"),

    behaviour: byId("birdBehaviour"),

    features: byId("birdIdentificationFeatures"),

    conservationStatus: byId("birdConservationStatus"),

    conservationNote: byId("birdConservationNote"),

    score: byId("similarityScore"),

    model: byId("modelName"),

    researchStatus: byId("researchStatus"),

    availability: byId("profileAvailability"),

    taxonomy: byId("taxonomySummary"),

    source: byId("profileSource"),

    sourcesSection: byId("sourcesSection"),

    sourceList: byId("sourceList"),

    qualityPanel: byId("qualityPanel"),

    qualityTitle: byId("qualityTitle"),

    qualityWarnings: byId("qualityWarnings"),

    note: byId("identificationNote")

};


let imageObjectUrl = null;


// ============================================================
// TEXT HELPERS
// ============================================================

function asDisplayText(
    value,
    fallback = "Not available in the current profile."
) {

    if (Array.isArray(value)) {

        const items = value
            .map((item) => String(item).trim())
            .filter(Boolean);

        return items.length
            ? items.join(" · ")
            : fallback;

    }

    if (typeof value === "number") {

        return String(value);

    }

    if (
        typeof value === "string" &&
        value.trim()
    ) {

        return value.trim();

    }

    return fallback;

}


function setText(
    element,
    value,
    fallback
) {

    if (!element) return;

    element.textContent = asDisplayText(
        value,
        fallback
    );

}


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
                new Error("Could not open saved image storage.")
            );

        };

    });

}


async function getSavedImage() {

    const db = await openImageDb();

    try {

        return await new Promise((resolve, reject) => {

            const tx = db.transaction(
                STORE_NAME,
                "readonly"
            );

            const request = tx.objectStore(
                STORE_NAME
            ).get("latest");

            request.onsuccess = () => {

                resolve(
                    request.result || null
                );

            };

            request.onerror = () => {

                reject(
                    request.error ||
                    new Error("Could not read the saved photograph.")
                );

            };

        });

    } finally {

        db.close();

    }

}


// ============================================================
// IMAGE QUALITY
// ============================================================

function renderImageQuality(quality) {

    if (!quality || !fields.qualityPanel) {

        return;

    }

    const warnings = Array.isArray(
        quality.warnings
    ) ? quality.warnings : [];


    fields.qualityPanel.hidden = false;


    if (!warnings.length) {

        fields.qualityTitle.textContent =
            "Basic photograph checks look acceptable";

        const item = document.createElement("li");

        item.textContent =
            "No obvious size, lighting, contrast or softness warning was detected.";

        fields.qualityWarnings.replaceChildren(item);

        return;

    }


    fields.qualityTitle.textContent =
        "A different photograph may produce a clearer match";


    const items = warnings.map((warning) => {

        const li = document.createElement("li");

        li.textContent = String(warning);

        return li;

    });


    fields.qualityWarnings.replaceChildren(
        ...items
    );

}


// ============================================================
// MODEL SCORE
// ============================================================

function renderScore(
    identification,
    modelName
) {

    const score = identification.similarity_score;

    if (
        typeof score === "number" &&
        Number.isFinite(score)
    ) {

        fields.score.textContent =
            `Cosine similarity: ${score.toFixed(3)} · not a probability`;

        fields.score.hidden = false;

    }


    if (modelName) {

        fields.model.textContent =
            `Model: ${modelName}`;

        fields.model.hidden = false;

    }

}


// ============================================================
// RESEARCH STATUS
// ============================================================

function renderResearchStatus(info) {

    const status = info.research_status || "unavailable";

    const researched = info.research_available === true;

    fields.researchStatus.dataset.state = status;

    if (status === "pending" || status === "running") {

        fields.researchStatus.dataset.state = "pending";

        fields.researchStatus.textContent =
            "Bird identified successfully · Gemini is preparing the species profile";

        fields.availability.textContent =
            "You can view the identification now. Habitat, diet, behaviour and other details are being researched.";

        fields.availability.hidden = false;

        return;
    }

    if (researched) {

        fields.researchStatus.dataset.state = "success";

        fields.researchStatus.textContent =
            info.search_grounding_enabled
                ? "Gemini research completed · Google Search grounding enabled"
                : "Gemini research completed · live web search disabled";

        fields.availability.textContent = info.cached
            ? "Species profile retrieved from the research cache."
            : "Species profile successfully generated by Gemini.";

        fields.availability.hidden = false;

        return;
    }

    fields.researchStatus.dataset.state = "fallback";

    fields.researchStatus.textContent =
        "Online research unavailable · showing local information where available";

    fields.availability.textContent =
        info.research_message ||
        "The identification is ready, but online species research could not be completed.";

    fields.availability.hidden = false;
}


// ============================================================
// TAXONOMY
// ============================================================

function renderTaxonomy(taxonomy) {

    if (
        !taxonomy ||
        typeof taxonomy !== "object"
    ) {

        return;

    }

    const parts = [];

    if (taxonomy.order) {

        parts.push(`Order: ${taxonomy.order}`);

    }

    if (taxonomy.family) {

        parts.push(`Family: ${taxonomy.family}`);

    }

    if (taxonomy.rank) {

        parts.push(`GBIF rank: ${taxonomy.rank}`);

    }


    if (parts.length) {

        fields.taxonomy.textContent =
            `GBIF taxonomy cross-check · ${parts.join(" · ")}`;

        fields.taxonomy.hidden = false;

    }

}


// ============================================================
// SOURCE LINKS
// ============================================================

function isSafeHttpUrl(value) {

    try {

        const parsed = new URL(value);

        return (
            parsed.protocol === "https:" ||
            parsed.protocol === "http:"
        );

    } catch {

        return false;

    }

}


function renderSources(info) {

    const rawSources = Array.isArray(info.sources)
        ? [...info.sources]
        : [];


    // Support older profiles with a single source_url.

    if (
        !rawSources.length &&
        isSafeHttpUrl(info.source_url)
    ) {

        rawSources.push({

            title: info.source_name ||
                "Species information source",

            url: info.source_url

        });

    }


    const unique = new Map();


    for (const source of rawSources) {

        if (
            !source ||
            !isSafeHttpUrl(source.url)
        ) {

            continue;

        }

        unique.set(
            source.url,
            {
                title: String(
                    source.title ||
                    new URL(source.url).hostname
                ),
                url: source.url
            }
        );

    }


    if (!unique.size) {

        return;

    }


    const links = [...unique.values()]
        .slice(0, 10)
        .map((source) => {

            const item = document.createElement("li");

            const link = document.createElement("a");

            link.href = source.url;

            link.target = "_blank";

            link.rel = "noopener noreferrer";

            link.textContent = source.title;

            item.appendChild(link);

            return item;

        });


    fields.sourceList.replaceChildren(...links);

    fields.sourcesSection.hidden = false;

}


// ============================================================
// PROFILE SOURCE
// ============================================================

function renderProfileSource(info) {

    if (info.provider) {

        fields.source.textContent =
            `${info.provider}${
                info.research_model
                    ? ` · ${info.research_model}`
                    : ""
            }`;

    } else if (info.source_name) {

        fields.source.textContent =
            `Profile source: ${info.source_name}`;

    } else {

        fields.source.textContent =
            "No external profile source was returned.";

    }

}


// ============================================================
// RESTORE IMAGE
// ============================================================

async function restoreUploadedImage() {

    try {

        const file = await getSavedImage();

        if (!file) {

            fields.image.alt =
                "The uploaded image is no longer stored in this browser.";

            fields.image.classList.add(
                "image-unavailable"
            );

            return;

        }


        imageObjectUrl = URL.createObjectURL(file);

        fields.image.src = imageObjectUrl;

    } catch (error) {

        console.warn(
            "Could not restore uploaded image:",
            error
        );

        fields.image.alt =
            "The uploaded photograph could not be restored.";

        fields.image.classList.add(
            "image-unavailable"
        );

    }

}


// ============================================================
// RENDER COMPLETE RESULT
// ============================================================

function renderResult(data) {

    const identification =
        data.identification || {};

    const info =
        data.bird_information || {};


    // Identification.

    setText(
        fields.common,
        identification.common_name,
        "Species name unavailable"
    );

    setText(
        fields.scientific,
        identification.scientific_name,
        "Scientific name unavailable"
    );

    setText(
        fields.description,
        info.description
    );


    // Species profile.

    setText(fields.habitat, info.habitat);

    setText(fields.diet, info.diet);

    setText(fields.distribution, info.distribution);

    setText(fields.family, info.family);

    setText(fields.behaviour, info.behaviour);

    setText(
        fields.features,
        info.identification_features
    );

    setText(
        fields.conservationStatus,
        info.conservation_status
    );

    setText(
        fields.conservationNote,
        info.conservation_note
    );


    // Upload metadata.

    setText(
        fields.filename,
        data.filename,
        "Uploaded photograph"
    );


    setText(
        fields.note,
        data.important_note,
        "This is the model's closest catalogue match, not a confirmed field identification."
    );


    // Supporting information.

    renderScore(
        identification,
        data.model
    );

    renderResearchStatus(info);

    renderTaxonomy(info.taxonomy);

    renderProfileSource(info);

    renderSources(info);

    renderImageQuality(data.image_quality);

}

const API_BASE_URL = "http://127.0.0.1:8000";

const sleep = (ms) =>
    new Promise(resolve => setTimeout(resolve, ms));


function updatePreparingStep(step, message) {

    const identification = document.getElementById(
        "stepIdentification"
    );

    const research = document.getElementById(
        "stepResearch"
    );

    const complete = document.getElementById(
        "stepComplete"
    );

    const messageElement = document.getElementById(
        "preparingMessage"
    );

    identification.className = "preparing-step done";

    identification.querySelector(".step-indicator").textContent = "✓";

    research.className = "preparing-step";

    complete.className = "preparing-step";

    if (step === "research") {

        research.classList.add("active");

        research.querySelector(".step-indicator").textContent = "…";

    }

    if (step === "complete") {

        research.classList.add("done");

        research.querySelector(".step-indicator").textContent = "✓";

        complete.classList.add("done");

        complete.querySelector(".step-indicator").textContent = "✓";

    }

    messageElement.textContent = message;

}


function revealReport() {

    const loader = document.getElementById("preparingPanel");

    const report = document.getElementById("reportContent");

    loader.hidden = true;

    report.hidden = false;

}


async function waitForResearch(data) {

    const researchId = data.research_id;

    if (!researchId) {

        return null;

    }

    const maximumWait = 8 * 60 * 1000;

    const started = Date.now();

    let consecutiveErrors = 0;

    while (Date.now() - started < maximumWait) {

        try {

            const response = await fetch(
                `${API_BASE_URL}/research/${encodeURIComponent(researchId)}`,
                { cache: "no-store" }
            );

            if (response.status === 404) {

                return null;

            }

            if (!response.ok) {

                throw new Error(`HTTP ${response.status}`);

            }

            const job = await response.json();

            consecutiveErrors = 0;

            if (job.status === "completed") {

                return job.result;

            }

            if (job.status === "failed") {

                return job.result || {
                    available: false,
                    message: "Species research could not be completed.",
                    research_status: "failed"
                };

            }

            updatePreparingStep(
                "research",
                "WingCLIP has identified the bird. Gemini is completing its species profile."
            );

        } catch (error) {

            consecutiveErrors++;

            console.error("Research status check failed:", error);

            if (consecutiveErrors >= 5) {

                return null;

            }

        }

        await sleep(2000);

    }

    return null;

}

// ============================================================
// INITIALIZE PAGE
// ============================================================

async function initializeResultPage() {

    const raw = sessionStorage.getItem(
        "birdIdentificationResult"
    );

    if (!raw) {

        document.getElementById("preparingPanel").hidden = true;

        document.getElementById("resultError").hidden = false;

        return;

    }

    let data;

    try {

        data = JSON.parse(raw);

    } catch (error) {

        console.error("Invalid result data:", error);

        return;

    }

    if (!data.identification) {

        return;

    }

    updatePreparingStep(
        "research",
        "WingCLIP has completed identification. Preparing the complete species report."
    );

    // The report stays hidden while Gemini research runs.
    const initialInfo = data.bird_information || {};

    if (data.research_id) {

        const research = await waitForResearch(data);

        if (research) {

            data.bird_information = {
                ...initialInfo,
                ...research,

                research_available: research.available === true,

                research_status:
                    research.research_status || "success",

                local_profile_available:
                    initialInfo.local_profile_available === true
            };

            if (!research.available && research.message) {

                data.bird_information.research_message =
                    research.message;

            }

        } else {

            data.bird_information = {
                ...initialInfo,

                research_status: "unavailable",

                research_available: false,

                research_message:
                    "Online research could not be retrieved. Showing the available local profile."
            };

        }

    }

    // Render everything at once.
    renderResult(data);

    // Save the final combined report.
    sessionStorage.setItem(
        "birdIdentificationResult",
        JSON.stringify(data)
    );

    await restoreUploadedImage();

    updatePreparingStep(
        "complete",
        "Your complete bird identification report is ready."
    );

    await sleep(450);

    revealReport();

}


// ============================================================
// CLEANUP
// ============================================================

window.addEventListener(
    "pagehide",
    () => {

        if (imageObjectUrl) {

            URL.revokeObjectURL(imageObjectUrl);

        }

    }
);


initializeResultPage();