import { Component } from '@angular/core';

@Component({
	selector: 'app-root',
	templateUrl: './app.component.html',
	styleUrls: ['./app.component.css']
})
export class AppComponent {

	title = 'demo';
	selected_item = 0;

	receiveItem($event) {
		this.selected_item = $event;
	}

	resetVis() {
		this.selected_item = 0;
	}
}
