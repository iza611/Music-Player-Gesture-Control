from presenter.app import RecordGesturesApp, PredictGestureLiveApp

if __name__ == "__main__":
    # app = RecordGesturesApp()
    app = PredictGestureLiveApp()
    app.run()