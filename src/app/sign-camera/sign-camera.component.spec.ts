import { ComponentFixture, TestBed } from '@angular/core/testing';

import { SignCameraComponent } from './sign-camera.component';

describe('SignCameraComponent', () => {
  let component: SignCameraComponent;
  let fixture: ComponentFixture<SignCameraComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [SignCameraComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(SignCameraComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
