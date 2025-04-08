document.addEventListener("DOMContentLoaded", function () {
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("fileInput");
    const fileNameDisplay = document.getElementById("file-name");
    const form = document.getElementById("prediction-form");
    const resultsContainer = document.getElementById("results-container");
    const predictButton = document.querySelector("button[type='submit']");

    // Function to update result values with proper mapping
    function updateResults(type, values) {
        const metrics = {
            traffic: document.querySelector(`#${type}-traffic .value`),
            congestion: document.querySelector(`#${type}-congestion .value`),
            speed: document.querySelector(`#${type}-speed .value`)
        };

        // Update each metric with its corresponding value
        if (metrics.traffic) metrics.traffic.textContent = formatValue(values[0]); // Traffic Volume
        if (metrics.congestion) metrics.congestion.textContent = formatValue(values[1]); // Congestion Level
        if (metrics.speed) metrics.speed.textContent = formatValue(values[2]); // Average Speed
    }

    // Helper function to format values
    function formatValue(value) {
        if (value === undefined || value === null) return '-';
        return typeof value === 'number' ? value.toFixed(2) : value;
    }

    // Reset results
    function resetResults() {
        ['actual', 'predicted'].forEach(type => {
            updateResults(type, ['-', '-', '-']);
        });
        resultsContainer.style.display = 'none';
    }

    dropZone.addEventListener("click", () => fileInput.click());

    dropZone.addEventListener("dragover", (event) => {
        event.preventDefault();
        dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));

    dropZone.addEventListener("drop", (event) => {
        event.preventDefault();
        dropZone.classList.remove("dragover");
        if (event.dataTransfer.files.length) {
            handleFileSelection(event.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", (event) => {
        if (event.target.files.length > 0) {
            handleFileSelection(event.target.files[0]);
        }
    });

    function handleFileSelection(file) {
        if (file) {
            fileNameDisplay.textContent = file.name;
            fileNameDisplay.style.display = 'block';
            resetResults();
        }
    }

    form.addEventListener("submit", async (event) => {
        event.preventDefault();
        const file = fileInput.files[0];
        if (!file) {
            alert("Please upload a CSV file.");
            return;
        }

        const formData = new FormData();
        formData.append("file", file);

        predictButton.disabled = true;
        predictButton.textContent = "Processing...";
        resetResults();

        try {
            const response = await fetch("/predict", {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                throw new Error("Prediction failed");
            }

            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }

            resultsContainer.style.display = 'block';
            
            // Update actual values if they exist
            if (data.actual && Array.isArray(data.actual)) {
                updateResults('actual', [
                    data.actual[0], // Traffic Volume
                    data.actual[1], // Congestion Level
                    data.actual[2]  // Average Speed
                ]);
            }
            
            // Update predicted values if they exist
            if (data.prediction && Array.isArray(data.prediction)) {
                updateResults('predicted', [
                    data.prediction[0], // Traffic Volume
                    data.prediction[1], // Congestion Level
                    data.prediction[2]  // Average Speed
                ]);
            }
            // For Plot 1: Traffic Volume
            if (data.plot_traffic) {
                document.getElementById('plot-traffic').src = `data:image/png;base64,${data.plot_traffic}`;
                document.getElementById('plot-traffic').style.display = 'block';
                }
    
            // For Plot 2: Congestion & Speed
            if (data.plot_cong_speed) {
            document.getElementById('plot-congestion-speed').src = `data:image/png;base64,${data.plot_cong_speed}`;
            document.getElementById('plot-congestion-speed').style.display = 'block';
            }
    
                

        } catch (error) {
            alert("Error processing the file: " + error.message);
            console.error("Error:", error);
        } finally {
            predictButton.disabled = false;
            predictButton.textContent = "Predict";
        }
    });
});