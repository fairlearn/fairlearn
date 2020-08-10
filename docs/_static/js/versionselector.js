function LoadVersions(targetfile, data) {
    console.log("Starting LoadVersions");
    listElement = document.getElementById('versionlist')
    for(const [k,v] of Object.entries(data)) {
      console.log("Processing "+k)
      var link = document.createElement("a");
      link.innerHTML = k;
      filepath = v.split("/");
      filepath[filepath.length-1] = targetfile;
      link.href=filepath.join("/");
      var li = document.createElement("li");
      li.appendChild(link)
      listElement.appendChild(li);
    }
    console.log("Completed LoadVersions")
  }