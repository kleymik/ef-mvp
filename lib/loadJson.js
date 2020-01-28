var data; // a global

d3.json("scatterData.json", function(error, json) {
                if (error) return console.warn(error);
                data = json;
                visualizeit();
});
