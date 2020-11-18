import { FairlearnModel, FairlearnView} from './fairlearnDashboard';

import {Application, IPlugin} from '@phosphor/application';
import {Widget} from '@phosphor/widgets';
import {IJupyterWidgetRegistry} from '@jupyter-widgets/base';
  
  const EXTENSION_ID = 'interpret-ml-widget:plugin';
  

  const fairlearnDashboardPlugin: IPlugin<Application<Widget>, void> = {
    id: EXTENSION_ID,
    requires: [IJupyterWidgetRegistry],
    activate: activateWidgetExtension,
    autoStart: true
  };
  
  export default fairlearnDashboardPlugin;
  
  const data = require('../package.json');
  /**
   * Activate the widget extension.
   */
  function activateWidgetExtension(app: Application<Widget>, registry: IJupyterWidgetRegistry): void {
    registry.registerWidget({
      name: 'interpret-ml-widget',
      version: data.version,
      exports: {
          FairlearnModel: FairlearnModel,
          FairlearnView: FairlearnView
      } as any
    });
  }