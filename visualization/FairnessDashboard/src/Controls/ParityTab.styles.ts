import { getTheme, IProcessedStyleSet, IStyle, mergeStyleSets, FontWeights, FontSizes } from 'office-ui-fabric-react';

export interface IParityTabStyles {
    iconClass: IStyle;
    itemsList: IStyle;
    frame: IStyle;
    main: IStyle;
    header: IStyle;
    textBody: IStyle;
}

export const ParityTabStyles: () => IProcessedStyleSet<IParityTabStyles> = () => {
    const theme = getTheme();
    return mergeStyleSets<IParityTabStyles>({
        iconClass: {
            fontSize: '20px',
            position: 'absolute',
            right: '10px',
            top: '10px',
        },
        itemsList: {
            overflowY: 'auto',
        },
        frame: {
            width: '750px',
            height: '100%',
        },
        main: {
            height: '100%',
            minWidth: '550px',
            flex: 1,
        },
        header: {
            color: theme.semanticColors.bodyText,
            fontWeight: FontWeights.semibold,
            margin: '26px 0',
        },
        textBody: {
            fontWeight: FontWeights.semilight,
            color: theme.semanticColors.bodyText,
            paddingTop: '12px',
            paddingBottom: '50px',
        },
    });
};
