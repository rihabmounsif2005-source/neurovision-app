document.querySelectorAll(".card").forEach(card => {
    card.addEventListener("mouseenter", () => {
        card.style.transform = "translateY(-5px)";
        card.style.transition = "0.3s";
    });

    card.addEventListener("mouseleave", () => {
        card.style.transform = "translateY(0px)";
    });
});
const fileInput = document.getElementById("fileInput");
const fileName = document.getElementById("fileName");

if (fileInput && fileName) {
    fileInput.addEventListener("change", function () {
        if (this.files.length > 0) {
            fileName.textContent = "Image sélectionnée : " + this.files[0].name;
        }
    });
}
function showSection(section, element) {

    document.querySelectorAll(".menu li").forEach(item => {
        item.classList.remove("active");
    });
    element.classList.add("active");

    const upload = document.querySelector(".upload-card");
    const preview = document.querySelector(".preview-card");
    const result = document.querySelector(".result-card");
    const exemples = document.querySelector(".examples-card");
    const historique = document.querySelector(".history-card");
    const stats = document.querySelector(".stats-card");
    const donut = document.querySelector(".donut-card");

    const allCards = [upload, preview, result, exemples, historique, stats, donut];

    allCards.forEach(card => {
        if (card) card.style.display = "none";
    });

    if (section === "accueil") {
        allCards.forEach(card => {
            if (card) card.style.display = "";
        });

        upload.style.gridColumn = "1 / 2";
        upload.style.gridRow = "1 / 2";

        preview.style.gridColumn = "2 / 3";
        preview.style.gridRow = "1 / 2";

        result.style.gridColumn = "3 / 4";
        result.style.gridRow = "1 / 2";
    }

    else if (section === "detection") {
        upload.style.display = "";
        preview.style.display = "";
        result.style.display = "";

        upload.style.gridColumn = "1 / 2";
        upload.style.gridRow = "1 / 2";

        preview.style.gridColumn = "2 / 3";
        preview.style.gridRow = "1 / 2";

        result.style.gridColumn = "3 / 4";
        result.style.gridRow = "1 / 2";
    }

    else if (section === "exemples") {
        exemples.style.display = "";
    }

    else if (section === "historique") {
        historique.style.display = "";
        stats.style.display = "";
        donut.style.display = "";
    }

    else if (section === "apropos") {
        alert("NeuroVision est une application pédagogique de détection des tumeurs cérébrales à partir d’images IRM.");
        allCards.forEach(card => {
            if (card) card.style.display = "";
        });
    }

    else if (section === "parametres") {
        alert("Paramètres : interface fixe, thème rouge/noir, analyse IRM.");
        allCards.forEach(card => {
            if (card) card.style.display = "";
        });
    }
}