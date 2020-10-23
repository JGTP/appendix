import { Component, OnInit, Input } from '@angular/core';
import { Router } from '@angular/router';


@Component({
	selector: 'app-vowl',
	templateUrl: './vowl.component.html',
	styleUrls: ['./vowl.component.css']
})
export class VowlComponent implements OnInit {

	@Input()
	selected_item: number;

	constructor(private router: Router) { }

	ngOnInit() {

	}



}
