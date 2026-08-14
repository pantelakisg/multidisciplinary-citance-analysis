"use strict";

const citanceInput = document.querySelector("#citance");
const modelSelect = document.querySelector("#model");
const analyzeButton = document.querySelector("#analyze");
const exampleButton = document.querySelector("#example");
const statusElement = document.querySelector("#status");
const cardsElement = document.querySelector("#cards");

const exampleCitances = [
  "Alongside this, variable practice is seen as a useful way of parameterising general motor programmes <CIT>.",
  "To measure HEL-specific antibodies, maxisorp plates (Nunc) were coated with 60 ml of recombinant HEL (10 mg/ml) diluted in NPP buffer (adapted from <CIT>).",
  "To provide programming model and runtime support, we extended an OpenMP implementation tailored for embedded multicore systems <CIT>, adding the features discussed in Sections IV-A and IV-B.",
  "While no significant risk was associated with dairy products in previous review <CIT>, our analysis resulted in increased CVD risk associated with dairy consumption.",
];

exampleButton.addEventListener("click", () => {
  const index = Math.floor(Math.random() * exampleCitances.length);
  citanceInput.value = exampleCitances[index];
  citanceInput.focus();
});

function resultCard(prediction) {
  const article = document.createElement("article");
  article.className = "result";

  const modelName = document.createElement("div");
  modelName.className = "model";
  modelName.textContent = prediction.model;
  article.append(modelName);

  const labels = document.createElement("div");
  labels.className = "labels";
  for (const task of ["semantics", "intent", "polarity"]) {
    const row = document.createElement("div");
    row.className = "label";

    const taskName = document.createElement("span");
    taskName.textContent = task[0].toUpperCase() + task.slice(1);

    const value = document.createElement("strong");
    value.textContent = prediction[task];
    if (prediction.confidence?.[task] !== undefined) {
      const confidence = document.createElement("small");
      confidence.className = "confidence";
      confidence.textContent = `${Math.round(prediction.confidence[task] * 100)}%`;
      value.append(" ", confidence);
    }

    row.append(taskName, value);
    labels.append(row);
  }
  article.append(labels);
  return article;
}

async function analyzeCitance() {
  analyzeButton.disabled = true;
  analyzeButton.textContent = "Running…";
  cardsElement.replaceChildren();
  statusElement.className = "status";
  statusElement.textContent = "Loading model and analyzing locally…";

  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        citance: citanceInput.value,
        model: modelSelect.value,
      }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Analysis failed.");
    }

    cardsElement.replaceChildren(...data.predictions.map(resultCard));
    statusElement.textContent = `Completed in ${data.elapsed_seconds} seconds.`;
  } catch (error) {
    statusElement.className = "status error";
    statusElement.textContent = error instanceof Error ? error.message : String(error);
  } finally {
    analyzeButton.disabled = false;
    analyzeButton.textContent = "Analyze citance";
  }
}

analyzeButton.addEventListener("click", analyzeCitance);
