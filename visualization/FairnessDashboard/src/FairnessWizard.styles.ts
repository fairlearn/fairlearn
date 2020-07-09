import { getTheme, IProcessedStyleSet, IStyle, mergeStyleSets } from 'office-ui-fabric-react';

export interface IFairnessWizardStyles {
    frame: IStyle;
    thinHeader: IStyle;
    headerLeft: IStyle;
    pivot: IStyle;
    body: IStyle;
    errorMessage: IStyle;
}

export const FairnessWizardStyles: () => IProcessedStyleSet<IFairnessWizardStyles> = () => {
    const theme = getTheme();
    return mergeStyleSets<IFairnessWizardStyles>({
        frame: {
            minHeight: '800px',
            minWidth: '800px',
        },
        thinHeader: {
            height: '36px',
            backgroundColor: theme.semanticColors.bodyBackground,
            color: theme.semanticColors.bodyText,
        },
        headerLeft: {
            padding: '20px',
        },
        pivot: {
            flex: 1,
            display: 'flex',
            flexDirection: 'column',
            padding: '30px 90px 0 82px',
        },
        body: {
            flex: 1,
            display: 'flex',
            flexDirection: 'column',
            backgroundColor: theme.semanticColors.bodyBackground,
        },
        errorMessage: {
            padding: '50px',
            fontSize: '18px',
        },
    });
};
