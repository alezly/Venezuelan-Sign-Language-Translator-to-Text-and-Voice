import { ComponentFixture, TestBed } from '@angular/core/testing';

import { SignCaptureComponent } from './sign-capture.component';

describe('SignCaptureComponent', () => {
  let component: SignCaptureComponent;
  let fixture: ComponentFixture<SignCaptureComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [SignCaptureComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(SignCaptureComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
