import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';
import { MatToolbarModule } from '@angular/material/toolbar';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { TimelineComponent } from './timeline/timeline.component';
import { VowlComponent } from './vowl/vowl.component';

@NgModule({
	declarations: [
		AppComponent,
		TimelineComponent,
		VowlComponent,
	],
	imports: [
		BrowserModule,
		AppRoutingModule,
		MatToolbarModule,
	],
	providers: [],
	bootstrap: [AppComponent]
})
export class AppModule { }
