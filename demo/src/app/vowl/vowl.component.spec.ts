import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { VowlComponent } from './vowl.component';

describe('VowlComponent', () => {
  let component: VowlComponent;
  let fixture: ComponentFixture<VowlComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ VowlComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(VowlComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
