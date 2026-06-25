const latestVersion = "v0.14"; // Do not include the patch version.
const versionPrefix = `./${latestVersion}`;

const versionedLinks = {
  gettingStartedLinkNavBar: "/quickstart.html",
  userGuideLinkNavBar: "/user_guide/index.html",
  apiRefLinkNavBar: "/api_reference/index.html",
  exampleNotebooksLinkNavBar: "/auto_examples/index.html",
  contribGuideLinkNavBar: "/contributor_guide/index.html",
  faqLinkNavBar: "/faq.html",
  aboutLinkNavBar: "/about/index.html",
  gettingStartedLinkHero: "/quickstart.html",
  fairnessGuideLinkHero: "/user_guide/fairness_in_machine_learning.html#fairness-of-ai-systems",
  caseStudyLinkCredit: "/auto_examples/plot_credit_loan_decisions.html",
  caseStudyLinkHealthcare: "/auto_examples/index.html",
  caseStudyLinkHiring: "/user_guide/mitigation/index.html",
  assessmentPathLink: "/user_guide/assessment/index.html",
  mitigationPathLink: "/user_guide/mitigation/index.html",
  datasetsPathLink: "/user_guide/datasets/index.html",
  userGuidePathLink: "/user_guide/index.html",
  apiRefPathLink: "/api_reference/index.html",
  contribGuidePathLink: "/contributor_guide/index.html",
  contribGuideLinkContribSection: "/contributor_guide/index.html",
};

function setLinks() {
  Object.entries(versionedLinks).forEach(([elementId, versionedPath]) => {
    const element = document.getElementById(elementId);
    if (element) {
      element.href = `${versionPrefix}${versionedPath}`;
    }
  });
}

function syncThemeToggleIcon(isDarkMode) {
  const themeToggleButton = document.getElementById("lightmode");
  if (!themeToggleButton) {
    return;
  }

  const lightModeIcon = themeToggleButton.querySelector(".icon-blue");
  const darkModeIcon = themeToggleButton.querySelector(".icon-white");

  if (lightModeIcon) {
    lightModeIcon.classList.toggle("active", !isDarkMode);
    lightModeIcon.classList.toggle("inactive", isDarkMode);
  }

  if (darkModeIcon) {
    darkModeIcon.classList.toggle("active", isDarkMode);
    darkModeIcon.classList.toggle("inactive", !isDarkMode);
  }

  themeToggleButton.setAttribute(
    "aria-label",
    isDarkMode ? "Switch to light mode" : "Switch to dark mode"
  );
}

function setTheme(isDarkMode) {
  document.body.classList.toggle("dark-theme", isDarkMode);
  syncThemeToggleIcon(isDarkMode);
}

function initializeThemeToggle() {
  const savedTheme = localStorage.getItem("theme");
  const prefersDarkMode = window.matchMedia("(prefers-color-scheme: dark)");
  const shouldUseDarkMode = savedTheme ? savedTheme === "dark-theme" : prefersDarkMode.matches;

  setTheme(shouldUseDarkMode);

  const themeToggleButton = document.getElementById("lightmode");
  if (!themeToggleButton) {
    return;
  }

  themeToggleButton.addEventListener("click", () => {
    const isDarkMode = !document.body.classList.contains("dark-theme");
    setTheme(isDarkMode);
    localStorage.setItem("theme", isDarkMode ? "dark-theme" : "");
  });

  if (!savedTheme) {
    prefersDarkMode.addEventListener("change", (event) => {
      setTheme(event.matches);
    });
  }
}

function initializeFooterYear() {
  const yearElement = document.getElementById("currentYear");
  if (yearElement) {
    yearElement.textContent = new Date().getFullYear().toString();
  }
}

function initializeMobileNavClose() {
  const navbar = document.getElementById("navbarResponsive");
  const toggleButton = document.querySelector(".navbar-toggler");
  if (!navbar || !toggleButton) {
    return;
  }

  const navLinks = navbar.querySelectorAll(".nav-link");
  navLinks.forEach((link) => {
    link.addEventListener("click", () => {
      if (window.innerWidth < 1200 && navbar.classList.contains("show")) {
        toggleButton.click();
      }
    });
  });
}

document.addEventListener("DOMContentLoaded", () => {
  setLinks();
  initializeThemeToggle();
  initializeFooterYear();
  initializeMobileNavClose();
});
