$(document).ready(function () {
  // executes when HTML-Document is loaded and DOM is ready
  console.log("document is ready");

  setLinks();

  $(".masthead-card").hover(
    function () {
      $(this).addClass('blue-card').css('cursor', 'pointer');
      $(".masthead-text", this).addClass('white');
      $(".icon-blue", this).addClass('inactive');
      $(".icon-blue", this).removeClass('active');
      $(".icon-white", this).addClass('active');
      $(".icon-white", this).removeClass('inactive');
    }, function () {
      $(this).removeClass('blue-card');
      $(".icon-blue", this).addClass('active');
      $(".icon-blue", this).removeClass('inactive');
      $(".icon-white", this).removeClass('active');
      $(".icon-white", this).addClass('inactive');
      $(".masthead-text", this).removeClass('white');
    }
  );


  $(".benefit-card").hover(
    function () {
      $(this).addClass('blue-card').css('cursor', 'pointer');
      $(".benefit-text", this).addClass('white');
      $(".icon-blue", this).addClass('inactive');
      $(".icon-blue", this).removeClass('active');
      $(".icon-white", this).addClass('active');
      $(".icon-white", this).removeClass('inactive');
    }, function () {
      $(this).removeClass('blue-card');
      $(".icon-blue", this).addClass('active');
      $(".icon-blue", this).removeClass('inactive');
      $(".icon-white", this).removeClass('active');
      $(".icon-white", this).addClass('inactive');
      $(".benefit-text", this).removeClass('white');
    }
  );



  $(".tc-1").hover(
    function () {
      $(this).addClass('white-card').css('cursor', 'pointer');
      $(".technique-text", this).addClass('dark');
      $(".technique-text", this).removeClass('white');
    }, function () {
    }
  );

  $(".tc-2").hover(
    function () {
      $(this).addClass('white-card').css('cursor', 'pointer');
      $(".technique-text", this).addClass('dark');
      $(".technique-text", this).removeClass('white');
      $(".global,.group,.feature").css("display", "none");
      $(".local").css("display", "block");
      globalFunctionOff();
    }, function () {
      $(this).removeClass('white-card');
      $(".technique-text", this).removeClass('dark');
      $(".technique-text", this).addClass('white');
      $(".global").css("display", "block");
      $(".local").css("display", "none");
      globalFunctionOn();
    }
  );

  $(".tc-3").hover(
    function () {
      $(this).addClass('white-card').css('cursor', 'pointer');
      $(".technique-text", this).addClass('dark');
      $(".technique-text", this).removeClass('white');
      $(".global,.local,.feature").css("display", "none");
      $(".group").css("display", "block");
      globalFunctionOff();
    }, function () {
      $(this).removeClass('white-card');
      $(".technique-text", this).removeClass('dark');
      $(".technique-text", this).addClass('white');
      $(".global").css("display", "block");
      $(".group").css("display", "none");
      globalFunctionOn();
    }
  );

  $(".tc-4").hover(
    function () {
      $(this).addClass('white-card').css('cursor', 'pointer');
      $(".technique-text", this).addClass('dark');
      $(".technique-text", this).removeClass('white');
      $(".global,.local,.group").css("display", "none");
      $(".feature").css("display", "block");
      globalFunctionOff();
    }, function () {
      $(this).removeClass('white-card');
      $(".technique-text", this).removeClass('dark');
      $(".technique-text", this).addClass('white');
      $(".global").css("display", "block");
      $(".feature").css("display", "none");
      globalFunctionOn();
    }
  );

  function globalFunctionOff() {
    $(".tc-1 .technique-text").removeClass("dark")
    $(".tc-1 .technique-text").addClass("white")
    $(".tc-1").removeClass("white-card")
  };

  function globalFunctionOn() {
    $(".tc-1 .technique-text").addClass('dark');
    $(".tc-1 .technique-text").removeClass('white');
    $(".tc-1").addClass("white-card")
  };

  $(".box-card-1").click(
    function () {
      console.log("box 1 clicked");
      $(".glass").removeClass('inactive');
      $(".black").removeClass('active');
    }
  );

  $(".box-card-2").click(
    function () {
      console.log("box 2 clicked");
      $(".glass").addClass('inactive');
      $(".black").addClass('active');
    }
  );


  $(document).click(function (event) {
    var clickover = $(event.target);
    var _opened = $(".navbar-collapse").hasClass("show");
    if (_opened === true && !clickover.hasClass("navbar-toggler")) {
      $(".navbar-toggler").click();
    }
  });


  $(document).ready(function () {
    // Check if the user has changed the theme before
    const savedTheme = localStorage.getItem('theme');

    // If no saved preference, check system preference
    if (savedTheme) {
      $("body").addClass(savedTheme);
    } else {
      const prefersDarkMode = window.matchMedia('(prefers-color-scheme: dark)').matches;
      if (prefersDarkMode) {
        $("body").addClass("dark-theme");
      }
    }

    $("#lightmode").click(function () {
      $("body").toggleClass("dark-theme");

      // Save the current theme
      if ($("body").hasClass("dark-theme")) {
        localStorage.setItem('theme', 'dark-theme');
      } else {
        localStorage.setItem('theme', '');
      }
    });
  });


});

async function setLinks() {
  // update this version with every release
  const latestVersion = 'v0.13'; // Do not include the patch
  const prefix = "./" + latestVersion;

  const contribGuideLink = prefix + "/contributor_guide/index.html";
  const exampleNotebooksLink = prefix + "/auto_examples/index.html";
  const apiRefLink = prefix + "/api_reference/index.html";
  const userGuideLink = prefix + "/user_guide/index.html";
  const gettingStartedLink = prefix + "/quickstart.html";
  const faqLink = prefix + "/faq.html";
  const aboutLink = prefix + "/about/index.html";

  document.getElementById('contribGuideLinkNavBar').href = contribGuideLink;
  document.getElementById('exampleNotebooksLinkNavBar').href = exampleNotebooksLink;
  document.getElementById('apiRefLinkNavBar').href = apiRefLink;
  document.getElementById('userGuideLinkNavBar').href = userGuideLink;
  document.getElementById('gettingStartedLinkNavBar').href = gettingStartedLink;
  document.getElementById('faqLinkNavBar').href = faqLink;
  document.getElementById('aboutLinkNavBar').href = aboutLink;

  document.getElementById('gettingStartedLinkMastHead').href = gettingStartedLink;
  document.getElementById('gettingStartedLinkBody').href = gettingStartedLink;
  document.getElementById('apiRefLinkBody').href = apiRefLink;
  document.getElementById('userGuideLinkBody').href = userGuideLink;
  document.getElementById('contribGuideLinkBody').href = contribGuideLink;
  document.getElementById('contribGuideLinkContribSection').href = contribGuideLink;
}
