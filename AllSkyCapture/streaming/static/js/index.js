  window.addEventListener("load", function() {
	var allInputs = document.querySelectorAll("input");
	allInputs.forEach((element) => {
		element.addEventListener("input", (event) => {
			data = {};
			allInputs.forEach((element) => {
				data[element.id] = parseInt(element.value) || element.checked;
			});
			fetch("/update", {
				method: "POST",
				headers: {
					'Content-Type': 'application/json'
				},
				body: JSON.stringify(data)
			});
		});
	});
	
	var toggles = [
		["#autoexposure", "#automaxexposure"],
		["#autogain", "#automaxgain"]
	];
	
	toggles.forEach((toggle) => {
		var element = document.querySelector(toggle[0]);
		var other = document.querySelector(toggle[1]);
		other.disabled = element.checked;
		
		element.addEventListener("click", (event) => {
			other.disabled = element.checked;
		});
	});
  });
