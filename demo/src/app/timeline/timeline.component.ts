import { Component, AfterContentInit, EventEmitter, Output } from '@angular/core';
import * as vis from 'vis';

@Component({
	selector: 'app-timeline',
	templateUrl: './timeline.component.html',
	styleUrls: ['./timeline.component.css']
})
export class TimelineComponent implements AfterContentInit {
	public selected_item = 0;

	@Output()
	itemEvent = new EventEmitter<number>();


	constructor() { }

	ngAfterContentInit() {
		var items = new vis.DataSet([
			{ id: 1, content: "Explosion", start: "2020-08-15T15:42" },
			{ id: 2, content: "First reports", start: "2020-08-15T16:08" },
			{ id: 3, content: "Further information", start: "2020-08-15T16:53" },
			{ id: 4, content: "Victim details", start: "2020-08-15T17:31" },
		]);

		var container = document.getElementById("visualization");
		var options = {
			editable: false,
		};
		var timeline = new vis.Timeline(container, items, options);
		let that = this;
		timeline.on("select", function(properties) {
			that.selected_item = properties.items[0];
			that.itemEvent.emit(that.selected_item);
		});



	}

}
