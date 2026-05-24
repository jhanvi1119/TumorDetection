window.addEventListener('scroll', function() {
    var header = document.getElementById('header');
    if (window.scrollY > 50) {
      header.classList.add('header-scrolled');
    } else {
      header.classList.remove('header-scrolled');
    }
  });

function scrollToTeam() {
  let targetElement = document.getElementsByTagName("footer")[0];
  targetElement.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function scrollToGoals() {
  let targetElement = document.getElementsByClassName("goals-container")[0];
  targetElement.scrollIntoView({ behavior: 'smooth', block: 'start' });
}
