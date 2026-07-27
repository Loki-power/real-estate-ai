/* Vercel Client-Side API Integration Script */
document.addEventListener("DOMContentLoaded", function() {
    const form = document.getElementById("prediction-form");
    if (!form) return;

    form.addEventListener("submit", async function(e) {
        e.preventDefault();
        
        const payload = {
            bedrooms: parseInt(document.getElementById("bedrooms").value),
            bathrooms: parseFloat(document.getElementById("bathrooms").value),
            sqft_living: parseInt(document.getElementById("sqft_living").value),
            sqft_lot: parseInt(document.getElementById("sqft_lot").value),
            floors: parseFloat(document.getElementById("floors").value),
            waterfront: parseInt(document.getElementById("waterfront").value),
            view: parseInt(document.getElementById("view").value),
            condition: parseInt(document.getElementById("condition").value),
            grade: parseInt(document.getElementById("grade").value),
            sqft_above: parseInt(document.getElementById("sqft_above").value),
            sqft_basement: parseInt(document.getElementById("sqft_basement").value),
            yr_built: parseInt(document.getElementById("yr_built").value),
            yr_renovated: parseInt(document.getElementById("yr_renovated").value),
            zipcode: parseInt(document.getElementById("zipcode").value),
            lat: 47.560,
            long: -122.213,
            sqft_living15: parseInt(document.getElementById("sqft_living").value),
            sqft_lot15: parseInt(document.getElementById("sqft_lot").value)
        };

        const resultBox = document.getElementById("result-box");
        const priceDisplay = document.getElementById("price-display");
        const categoryBadge = document.getElementById("category-badge");
        const explanationText = document.getElementById("explanation-text");

        try {
            priceDisplay.innerText = "Calculating...";
            const response = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            if (!response.ok) throw new Error("Prediction API Error");

            const data = await response.json();

            priceDisplay.innerText = `$${data.predicted_price.toLocaleString(undefined, {minimumFractionDigits: 2})}`;
            categoryBadge.innerText = data.category;
            categoryBadge.style.backgroundColor = data.badge_color;
            explanationText.innerText = data.explanation;

        } catch (err) {
            priceDisplay.innerText = "Error";
            explanationText.innerText = `Could not generate prediction: ${err.message}`;
        }
    });
});
